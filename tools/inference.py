from mmseg.apis import MMSegInferencer
# hack to load custom models (looks like outdated)
#import mmseg.models.backbones.mobile_sam_vit
#import mmseg.models.backbones.sam_vit
#import mmseg.engine.hooks.force_test_loop_hook
#mport mmseg.engine.hooks.best_model_testing_hook

from pathlib import Path
from argparse import ArgumentParser

import numpy as np
from numpy.lib import format as npy_format
import pathlib
import time
import zipfile


def _write_npz_member(zf, name, array):
    """Stream a single array into an open npz-style zip as <name>.npy.

    Matches the on-disk layout np.savez produces, so np.load() reads it back —
    but we write one array at a time instead of holding every mask in RAM.
    """
    with zf.open(name + ".npy", "w", force_zip64=True) as fid:
        npy_format.write_array(fid, np.asarray(array), allow_pickle=False)

import torch
_original_torch_load = torch.load
torch.load = lambda *args, **kwargs: _original_torch_load(*args, **{**kwargs, 'weights_only': False})



class FPSLogger:
    def __init__(self):
        self.tottime = 0.0
        self.count = 0
        self.last_record = 0.0
        self.last_print = time.time()
        self.interval = 3

    def start_record(self):
        self.last_record = time.time()

    def end_record(self):
        self.tottime += time.time() - self.last_record
        self.count += 1
        self.print_fps()

    def print_fps(self):
        if time.time() - self.last_print > self.interval:
            print(f"Inference running at {self.count / self.tottime:.3f} FPS")
            self.last_print = time.time()


def main(config):
    fps_logger = FPSLogger()
    inference = MMSegInferencer(
        model=config.config,
        weights=config.checkpoint,
    )
    if config.silent:
        inference.show_progress = False

    images : Path = config.images_dir
    output_dir: Path = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Only process the requested images. `images` holds the whole dataset, but a
    # prediction run targets just the tasks in --input_npz. That npz is keyed by
    # image name and carries each task's ground-truth mask (or an empty array if
    # unannotated), so metrics can later be computed against it. We take the
    # image names from its keys. Without it we fall back to the whole folder.
    if config.input_npz:
        with np.load(config.input_npz) as gt:
            names = sorted({k[: -len("__shape")] for k in gt.files if k.endswith("__shape")})
        image_paths = [images / name for name in names]
        print(f"Processing {len(image_paths)} images from {config.input_npz}")
    else:
        image_paths = list(images.glob("**/*.jpg"))

    start_time = time.time()
    if config.npz:
        # Stream every predicted mask into a single packed .npz, one image at a
        # time, so peak RAM is one mask — not all of them. Keys mirror the
        # training format (<name>__packed / <name>__shape) so the predict side
        # can read them back and convert to RLE lazily. We also store a per-image
        # confidence score (<name>__score) for the prediction.
        npz_path = pathlib.Path(output_dir) / "prediction.npz"
        with zipfile.ZipFile(
            npz_path, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True
        ) as zf:
            for p in image_paths:
                if not p.exists():
                    print(f"Skipping missing image: {p}")
                    continue
                try:
                    fps_logger.start_record()
                    # return_datasamples gives access to seg_logits for the score.
                    ds = inference(str(p), return_datasamples=True)
                    seg = ds.pred_sem_seg.data.squeeze(0)          # (H, W) class ids
                    mask = (seg > 0).to(torch.uint8).cpu().numpy()  # binary 0/1

                    # Confidence = mean max-softmax probability over the predicted
                    # foreground (or the whole image if nothing was predicted).
                    try:
                        probs = ds.seg_logits.data.softmax(dim=0)
                        conf = probs.max(dim=0).values             # (H, W)
                        fg = seg > 0
                        score = float(conf[fg].mean() if fg.any() else conf.mean())
                    except Exception:
                        score = 1.0

                    # Key by full filename (<sha512>.jpg) so the predict side can
                    # match it to a task via image_path.split('/')[-1].
                    _write_npz_member(zf, f"{p.name}__packed", np.packbits(mask.ravel()))
                    _write_npz_member(zf, f"{p.name}__shape", np.array(mask.shape, dtype=np.int32))
                    _write_npz_member(zf, f"{p.name}__score", np.array([score], dtype=np.float32))
                    fps_logger.end_record()

                except Exception as e:
                    print(f"Failed with {p}. {e}")

    else:
        for p in images.glob("**/*.jpg"):
            print(p)
            inference(
                str(p),
                out_dir=str(output_dir)
            )
    print(f"Inference time: {round(time.time() - start_time, 2)} s.")



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("config")
    parser.add_argument("images_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--npz",
        action="store_true",
        help="save all masks streamed into a single packed prediction.npz")
    parser.add_argument(
        "--input_npz",
        type=str,
        default=None,
        help="path to an input .npz keyed by image name (with each task's GT mask "
             "or an empty array); its keys select which images to process. "
             "If omitted, all .jpg in images_dir are used")
    parser.add_argument(
        "--silent",
        action="store_true",
        help="suppress progress bars and verbose output")
    config = parser.parse_args()
    main(config)
