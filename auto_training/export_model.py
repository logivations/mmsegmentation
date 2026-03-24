import argparse
import os

import torch
import torch.nn as nn
import numpy as np
from mmengine.config import Config
from mmseg.apis import init_model

# Patch torch.load for PyTorch 2.6+ (weights_only=True breaks numpy in checkpoints)
_original_torch_load = torch.load
torch.load = lambda *args, **kwargs: _original_torch_load(*args, **{**kwargs, 'weights_only': False})

try:
    import onnx
    import onnxruntime as rt
except ImportError as e:
    raise ImportError(f'Please install onnx and onnxruntime first. {e}')


import torch.nn.functional as F


def _replace_adaptive_pools(model, dummy_input):
    """Replace all AdaptiveAvgPool2d with AvgPool2d with static kernel/stride.

    During ONNX tracing, input.shape[i] is recorded as aten::size ops, making
    kernel sizes computed from it non-constant. ONNX AveragePool requires
    constant kernel_shape attributes. This function replaces modules BEFORE
    tracing so kernel sizes become Python ints (constants) in the ONNX graph.
    """
    hooks = []
    # {module_id: (parent_module, attr_name, new_module)}
    replacements = {}
    module_refs = {}

    def make_hook(module):
        def hook(m, input, output):
            h_in = int(input[0].shape[2])
            w_in = int(input[0].shape[3])
            output_size = m.output_size
            if isinstance(output_size, int):
                h_out = w_out = output_size
            elif output_size is None:
                h_out = w_out = 1
            else:
                h_out, w_out = int(output_size[0]), int(output_size[1])

            if h_out == 1 and w_out == 1:
                new_pool = nn.AvgPool2d(kernel_size=(h_in, w_in), stride=(h_in, w_in))
            else:
                stride_h = h_in // h_out
                stride_w = w_in // w_out
                kernel_h = h_in - (h_out - 1) * stride_h
                kernel_w = w_in - (w_out - 1) * stride_w
                new_pool = nn.AvgPool2d(kernel_size=(kernel_h, kernel_w), stride=(stride_h, stride_w))

            replacements[id(m)] = new_pool
        return hook

    for name, module in model.named_modules():
        if isinstance(module, nn.AdaptiveAvgPool2d):
            hook = module.register_forward_hook(make_hook(module))
            hooks.append(hook)
            module_refs[id(module)] = name

    with torch.no_grad():
        model(dummy_input)

    for hook in hooks:
        hook.remove()

    named_modules_dict = dict(model.named_modules())
    for mod_id, new_pool in replacements.items():
        name = module_refs[mod_id]
        parts = name.rsplit('.', 1)
        if len(parts) == 2:
            parent = named_modules_dict[parts[0]]
            setattr(parent, parts[1], new_pool)
        else:
            setattr(model, name, new_pool)
        print(f'  Replaced AdaptiveAvgPool2d at "{name}" -> AvgPool2d(kernel={new_pool.kernel_size}, stride={new_pool.stride})')


class ExportWrapper(nn.Module):
    """Wraps an mmseg segmentor for ONNX export, replicating mmdeploy's
    rewriter logic.

    Args:
        model: An mmseg segmentor (e.g., EncoderDecoder).
        with_argmax: Whether to apply argmax to the output.
            mmdeploy defaults to True, but downstream postprocessing
            (e.g., ORTSegmentationDetector) typically applies argmax
            itself, so this defaults to False to avoid double application.
    """

    def __init__(self, model, with_argmax=False):
        super().__init__()
        self.model = model
        self.with_argmax = with_argmax

        if hasattr(model, 'num_stages'):
            self.align_corners = model.decode_head[-1].align_corners
        else:
            self.align_corners = model.decode_head.align_corners

    def forward(self, inputs):

        img_shape = inputs.shape[2:]
        x = self.model.extract_feat(inputs)

        if hasattr(self.model, 'num_stages'):
            out = self.model.decode_head[0].forward(x)
            for i in range(1, self.model.num_stages - 1):
                out = self.model.decode_head[i].forward(x, out)
            seg_logits = self.model.decode_head[-1].forward(x, out)
        else:
            seg_logits = self.model.decode_head.forward(x)

        seg_logits = F.interpolate(
            seg_logits,
            size=img_shape,
            mode='bilinear',
            align_corners=self.align_corners)

        if self.with_argmax:
            if seg_logits.shape[1] == 1:
                seg_logits = seg_logits.sigmoid()
                seg_pred = (seg_logits > self.model.decode_head.threshold).to(
                    torch.int64)
            else:
                seg_pred = seg_logits.argmax(dim=1, keepdim=True)
            return seg_pred

        return seg_logits


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert MMSegmentation models to ONNX')
    parser.add_argument('config', help='config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--show', action='store_true', help='show onnx graph')
    parser.add_argument('--output-file', type=str, default='tmp.onnx')
    parser.add_argument('--opset-version', type=int, default=16)
    parser.add_argument(
        '--verify',
        action='store_true',
        help='verify the onnx model output against pytorch output')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[1, 3, 512, 512],
        help='input size')
    parser.add_argument(
        '--static-batch',
        action='store_true',
        help='disable dynamic batch size export')
    args = parser.parse_args()
    return args


def pytorch2onnx(model, input_shape, opset_version, output_file, verify, show, dynamic_batch=False):
    """Export PyTorch model to ONNX format."""
    model.eval()

    dummy_input = torch.randn(input_shape)

    # Replace AdaptiveAvgPool2d with static AvgPool2d BEFORE tracing.
    # This is required because ONNX AveragePool needs constant kernel sizes,
    # but computing them from input.shape inside the trace produces dynamic ops.
    print('Replacing AdaptiveAvgPool2d with static AvgPool2d...')
    _replace_adaptive_pools(model, dummy_input)

    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'},
        }

    # Export to ONNX
    print(f'Exporting model to ONNX with input shape: {input_shape}, dynamic_batch={dynamic_batch}')
    torch.onnx.export(
        model,
        dummy_input,
        output_file,
        input_names=['input'],
        output_names=['output'],
        opset_version=opset_version,
        dynamic_axes=dynamic_axes,
        keep_initializers_as_inputs=False
    )

    print(f'Successfully exported ONNX model: {output_file}')

    # Verify the exported model
    if verify:
        print('Verifying ONNX model...')
        onnx_model = onnx.load(output_file)
        onnx.checker.check_model(onnx_model)

        # Compare outputs
        with torch.no_grad():
            pytorch_output = model(dummy_input).numpy()

        sess = rt.InferenceSession(output_file)
        onnx_output = sess.run(None, {'input': dummy_input.numpy()})[0]

        if np.allclose(pytorch_output, onnx_output, rtol=1e-3, atol=1e-5):
            print('ONNX model verified: outputs match PyTorch!')
        else:
            print('Warning: ONNX outputs differ from PyTorch outputs')
            print(f'Max difference: {np.max(np.abs(pytorch_output - onnx_output))}')

    if show:
        onnx_model = onnx.load(output_file)
        print(onnx.helper.printable_graph(onnx_model.graph))


def export_model(config_path, checkpoint_path, output_dir=None, input_shape=(1, 3, 512, 512), opset_version=16, dynamic_batch=True):
    """Export a trained MMSegmentation model to ONNX format.

    Args:
        config_path: Path to the config file or Config object
        checkpoint_path: Path to the checkpoint file
        output_dir: Directory to save the exported model. If None, uses config's work_dir
        input_shape: Input shape for the model (default: (1, 3, 512, 512))
        opset_version: ONNX opset version (default: 17)

    Returns:
        str: Path to the exported ONNX model
    """
    if isinstance(config_path, Config):
        cfg = config_path
    else:
        cfg = Config.fromfile(config_path)

    # build the model using init_model which handles registration
    model = init_model(cfg, checkpoint_path, device='cpu')

    model = ExportWrapper(model)

    # Determine output path
    if output_dir is None:
        output_dir = os.path.join(cfg.work_dir)
    os.makedirs(output_dir, exist_ok=True)
    model_output_path = os.path.join(output_dir, "segm_model.onnx")

    pytorch2onnx(
        model,
        input_shape,
        opset_version=opset_version,
        show=False,
        output_file=model_output_path,
        verify=False,
        dynamic_batch=dynamic_batch)

    print(f"Model exported successfully to: {model_output_path}")
    return model_output_path

# python3 -m auto_training.export_model /path/to/custom_vit_uper.py /path/to/best_val_mDice_iter_111.pth
if __name__ == '__main__':
    args = parse_args()

    export_model(
        args.config,
        args.checkpoint,
        input_shape=tuple(args.shape),
        opset_version=args.opset_version,
        dynamic_batch=not args.static_batch
    )
