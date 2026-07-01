import os.path as osp

import numpy as np

from mmseg.datasets.basesegdataset import BaseSegDataset
from mmseg.registry import DATASETS


@DATASETS.register_module()
class CustomMountedEmpty(BaseSegDataset):

    METAINFO = dict(
        classes=[],
        palette=(),
    )

    def __init__(
        self,
        target_class_map=None,
        classes: list = None,
        img_suffix: str = '.jpg',
        seg_map_suffix: str = '.png',
        ann_npz_file: str = None,
        **kwargs
    ) -> None:
        print(f"Found classes: {classes}")
        self.METAINFO["classes"] = []
        for cls in classes:
            if cls in target_class_map:
                if target_class_map[cls] is not None:
                    self.METAINFO["classes"].append(target_class_map[cls])
            else:
                self.METAINFO["classes"].append(cls)

        print(f"Classes for training: {self.METAINFO['classes']}")
        print(f"Target_class_map: {target_class_map}")

        self._ann_npz_file = ann_npz_file
        self._npz_path = None
        self._available_keys = None
        self._packed_format = True

        super().__init__(img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)

    def _index_mask_npz(self):
        """Read ONLY the mask key index from the NPZ — no mask arrays are loaded.

        Masks are read lazily, one per sample, by ``LoadAnnotationsFromCache``,
        so a 120k-mask split costs O(keys) of RAM here instead of O(decoded
        masks) — which is what used to OOM training.
        """
        if self._ann_npz_file is None:
            return
        npz_path = self._ann_npz_file
        if not osp.isabs(npz_path):
            npz_path = osp.join(self.data_root, npz_path)
        self._npz_path = npz_path
        print(f"Indexing NPZ annotations from {npz_path}...")
        with np.load(npz_path) as npz:
            names = npz.files  # zip central directory only — no arrays loaded
        packed = {n[:-7] for n in names if n.endswith('__shape')}
        if packed:
            self._available_keys = packed
            self._packed_format = True
        else:
            self._available_keys = set(names)
            self._packed_format = False
        print(f"Indexed {len(self._available_keys)} masks from NPZ")

    def load_data_list(self):
        self._index_mask_npz()
        data_list = super().load_data_list()
        if self._available_keys is None:
            return data_list
        result = []
        for d in data_list:
            key = osp.basename(d['img_path'])
            if key in self._available_keys:
                d['seg_map_key'] = key
                d['ann_npz_file'] = self._npz_path
                d['seg_map_packed'] = self._packed_format
                result.append(d)
        return result

