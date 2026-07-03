import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from WFlib.tools import data_processor
from comparison.common import dataset_dir, load_config, load_df_class, progress, result_subdir, write_json


def load_label_names(dataset_path):
    label_map = {}
    path = dataset_path / "label_map.json"
    if path.exists():
        import json
        with path.open("r", encoding="utf-8") as fp:
            raw = json.load(fp)
        label_map = {int(v): k for k, v in raw.items()}
    return label_map


def evaluate(cfg, dataset, checkpoint, device_name):
    seq_len = int(cfg.get("seq_len", 5000))
    batch_size = int(cfg.get("eval", {}).get("batch_size", 256))
    num_workers = int(cfg.get("eval", {}).get("num_workers", 0))
    dataset_path = dataset_dir(cfg, dataset)
    test_path = dataset_path / "test.npz"
    X, y = data_processor.load_data(str(test_path), cfg.get("feature", "DIR"), seq_len, 1)
    num_classes = int(y.max().item()) + 1
    loader = data_processor.load_iter(X, y, batch_size, False, num_workers)

    device = torch.device(device_name)
    model = load_df_class()(num_classes)
    model.load_state_dict(torch.load(str(checkpoint), map_location="cpu"))
    model.to(device)
    model.eval()
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    y_true, y_pred = [], []
    loss_sum = 0.0
    count = 0
    with torch.no_grad():
        for cur_X, cur_y in progress(loader, desc=f"{dataset} eval", unit="batch"):
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
        "model": cfg.get("model", "DF"),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 6),
        "loss": round(float(loss_sum / max(count, 1)), 6),
        "num_test_samples": int(count),
        "seq_length": seq_len,
        "batch_size": batch_size,
        "checkpoint": str(checkpoint),
    }

    out_dir = result_subdir(cfg, "metrics", dataset)
    write_json(out_dir / "_metrics.json", metrics)

    labels = list(range(num_classes))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    label_names = load_label_names(dataset_path)
    with (out_dir / "confusion_matrix.csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["true_label"] + [label_names.get(i, str(i)) for i in labels])
        for idx, row in enumerate(matrix):
            writer.writerow([label_names.get(idx, str(idx))] + [int(v) for v in row])
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate DF checkpoint with Accuracy, Macro-F1, and loss.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    metrics = evaluate(cfg, args.dataset, Path(args.checkpoint), args.device)
    print(metrics)
    return 0


if __name__ == "__main__":
    sys.exit(main())
