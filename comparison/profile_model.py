import argparse
import sys

import torch
from torch import nn

from WFlib.tools import data_processor
from comparison.common import instantiate_model, load_config, result_subdir, write_json
from comparison.feature_builders import model_data_dir
from comparison.model_registry import model_config


def infer_num_classes(data_dir):
    data = torch.from_numpy(__import__("numpy").load(data_dir / "train.npz")["y"])
    return int(data.max().item()) + 1


def estimate_dense_flops(model, sample_input):
    flops = {"value": 0}
    hooks = []

    def conv1d_hook(module, inputs, output):
        batch, out_channels, out_len = output.shape[:3]
        kernel = module.kernel_size[0]
        in_channels = module.in_channels
        groups = module.groups
        flops["value"] += int(batch * out_channels * out_len * (in_channels // groups) * kernel * 2)
        if module.bias is not None:
            flops["value"] += int(batch * out_channels * out_len)

    def conv2d_hook(module, inputs, output):
        batch, out_channels, out_h, out_w = output.shape[:4]
        kernel_h, kernel_w = module.kernel_size
        in_channels = module.in_channels
        groups = module.groups
        flops["value"] += int(batch * out_channels * out_h * out_w * (in_channels // groups) * kernel_h * kernel_w * 2)
        if module.bias is not None:
            flops["value"] += int(batch * out_channels * out_h * out_w)

    def linear_hook(module, inputs, output):
        batch = output.reshape(-1, output.shape[-1]).shape[0]
        flops["value"] += int(batch * module.in_features * module.out_features * 2)
        if module.bias is not None:
            flops["value"] += int(batch * module.out_features)

    for module in model.modules():
        if isinstance(module, nn.Conv1d):
            hooks.append(module.register_forward_hook(conv1d_hook))
        elif isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(conv2d_hook))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))

    model.eval()
    with torch.no_grad():
        model(sample_input)
    for hook in hooks:
        hook.remove()
    return flops["value"]


def profile(cfg, dataset, model_name):
    model_cfg = model_config(model_name)
    data_dir = model_data_dir(cfg, dataset, model_name)
    seq_len = int(model_cfg["seq_len"])
    X, y = data_processor.load_data(str(data_dir / "test.npz"), model_cfg["feature"], seq_len, 1)
    num_classes = int(y.max().item()) + 1 if len(y) else infer_num_classes(data_dir)
    sample_input = X[:1].cpu()
    model = instantiate_model(model_cfg, num_classes, num_tabs=1).cpu()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    dense_flops = estimate_dense_flops(model, sample_input)
    result = {
        "dataset": dataset,
        "model": model_name,
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "active_params_estimated": int(total_params),
        "active_param_ratio": 1.0,
        "dense_equivalent_flops": int(dense_flops),
        "effective_flops": int(dense_flops),
        "flops_method": "forward hooks over Conv1d, Conv2d, and Linear multiply-adds for batch_size=1; normalization/activation/pooling/attention matmul overhead omitted",
        "seq_length": seq_len,
        "input_shape": list(sample_input.shape),
        "conditional_compute": False,
    }
    write_json(result_subdir(cfg, "profiles", dataset, model_name) / "_profile.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Profile one WFlib model.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    print(profile(cfg, args.dataset, args.model))
    return 0


if __name__ == "__main__":
    sys.exit(main())
