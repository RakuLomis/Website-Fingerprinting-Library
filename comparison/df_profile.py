import argparse
import sys
from pathlib import Path

import torch
from torch import nn

from comparison.common import dataset_dir, load_config, load_df_class, result_subdir, write_json


def infer_num_classes(dataset_path):
    import json
    with (dataset_path / "label_map.json").open("r", encoding="utf-8") as fp:
        return len(json.load(fp))


def estimate_dense_flops(model, seq_len):
    flops = {"value": 0}
    hooks = []

    def conv_hook(module, inputs, output):
        batch = output.shape[0]
        out_channels = output.shape[1]
        out_len = output.shape[2]
        kernel = module.kernel_size[0]
        in_channels = module.in_channels
        groups = module.groups
        flops["value"] += int(batch * out_channels * out_len * (in_channels // groups) * kernel * 2)
        if module.bias is not None:
            flops["value"] += int(batch * out_channels * out_len)

    def linear_hook(module, inputs, output):
        batch = output.shape[0]
        flops["value"] += int(batch * module.in_features * module.out_features * 2)
        if module.bias is not None:
            flops["value"] += int(batch * module.out_features)

    for module in model.modules():
        if isinstance(module, nn.Conv1d):
            hooks.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))

    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 1, seq_len))
    for hook in hooks:
        hook.remove()
    return flops["value"]


def profile(cfg, dataset):
    seq_len = int(cfg.get("seq_len", 5000))
    num_classes = infer_num_classes(dataset_dir(cfg, dataset))
    model = load_df_class()(num_classes)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    dense_flops = estimate_dense_flops(model, seq_len)
    result = {
        "dataset": dataset,
        "model": cfg.get("model", "DF"),
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "active_params_estimated": int(total_params),
        "active_param_ratio": 1.0,
        "dense_equivalent_flops": int(dense_flops),
        "effective_flops": int(dense_flops),
        "flops_method": "forward hooks over Conv1d and Linear multiply-adds for batch_size=1; BatchNorm/activation/pooling/dropout omitted",
        "seq_length": seq_len,
        "conditional_compute": False,
    }
    write_json(result_subdir(cfg, "profiles", dataset) / "_profile.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Profile DF parameter counts and dense-equivalent FLOPs.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    result = profile(cfg, args.dataset)
    print(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
