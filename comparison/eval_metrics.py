import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from WFlib.tools import data_processor
from comparison.common import instantiate_model, load_config, progress, result_subdir, write_json
from comparison.feature_builders import base_dataset_dir, model_data_dir
from comparison.model_registry import model_config


def load_label_names(dataset_path):
    path = dataset_path / "label_map.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fp:
        raw = json.load(fp)
    return {int(v): k for k, v in raw.items()}


def evaluate(cfg, dataset, model_name, checkpoint, device_name):
    model_cfg = model_config(model_name)
    if model_cfg.get("loss") != "CrossEntropyLoss":
        raise NotImplementedError(f"{model_name} is not supported by generic logits evaluation yet.")

    seq_len = int(model_cfg["seq_len"])
    batch_size = int(model_cfg.get("eval_batch_size", cfg.get("eval", {}).get("batch_size", 256)))
    num_workers = int(cfg.get("eval", {}).get("num_workers", 0))
    data_dir = model_data_dir(cfg, dataset, model_name)
    X, y = data_processor.load_data(str(data_dir / "test.npz"), model_cfg["feature"], seq_len, 1)
    num_classes = int(y.max().item()) + 1
    loader = data_processor.load_iter(X, y, batch_size, False, num_workers)

    device = torch.device(device_name)
    model = instantiate_model(model_cfg, num_classes, num_tabs=1)
    model.load_state_dict(torch.load(str(checkpoint), map_location="cpu"))
    model.to(device)
    model.eval()
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    y_true, y_pred = [], []
    loss_sum = 0.0
    count = 0
    with torch.no_grad():
        for cur_X, cur_y in progress(loader, desc=f"{dataset}/{model_name} eval", unit="batch"):
            cur_X = cur_X.to(device)
            cur_y = cur_y.to(device)
            logits = model(cur_X)
            loss_sum += float(criterion(logits, cur_y).cpu().item())
            pred = torch.argmax(logits, dim=1)
            y_true.append(cur_y.cpu().numpy())
            y_pred.append(pred.cpu().numpy())
            count += int(cur_y.numel())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    metrics = {
        "dataset": dataset,
        "model": model_name,
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 6),
        "loss": round(float(loss_sum / max(count, 1)), 6),
        "num_test_samples": int(count),
        "seq_length": seq_len,
        "batch_size": batch_size,
        "checkpoint": str(checkpoint),
    }

    out_dir = result_subdir(cfg, "metrics", dataset, model_name)
    write_json(out_dir / "_metrics.json", metrics)
    labels = list(range(num_classes))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    label_names = load_label_names(base_dataset_dir(cfg, dataset))
    with (out_dir / "confusion_matrix.csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["true_label"] + [label_names.get(i, str(i)) for i in labels])
        for idx, row in enumerate(matrix):
            writer.writerow([label_names.get(idx, str(idx))] + [int(v) for v in row])
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate one WFlib model with Accuracy, Macro-F1, and loss.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    print(evaluate(cfg, args.dataset, args.model, Path(args.checkpoint), args.device))
    return 0


if __name__ == "__main__":
    sys.exit(main())
