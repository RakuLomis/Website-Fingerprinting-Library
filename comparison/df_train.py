import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

from WFlib.tools import data_processor
from comparison.common import checkpoint_dir, dataset_dir, load_config, load_df_class, progress, write_json


def train(cfg, dataset, device_name):
    seed = int(cfg.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_cfg = cfg.get("train", {})
    seq_len = int(cfg.get("seq_len", 5000))
    batch_size = int(train_cfg.get("batch_size", 128))
    num_workers = int(train_cfg.get("num_workers", 10))
    epochs = int(train_cfg.get("epochs", 30))
    learning_rate = float(train_cfg.get("learning_rate", 2e-3))
    optimizer_name = train_cfg.get("optimizer", "Adamax")
    save_name = train_cfg.get("save_name", "max_f1")

    if device_name.startswith("cuda"):
        assert torch.cuda.is_available(), f"The specified device {device_name} does not exist"
    device = torch.device(device_name)

    data_dir = dataset_dir(cfg, dataset)
    train_X, train_y = data_processor.load_data(str(data_dir / "train.npz"), cfg.get("feature", "DIR"), seq_len, 1)
    valid_X, valid_y = data_processor.load_data(str(data_dir / "val.npz"), cfg.get("feature", "DIR"), seq_len, 1)
    num_classes = int(train_y.max().item()) + 1

    train_iter = data_processor.load_iter(train_X, train_y, batch_size, True, num_workers)
    valid_iter = data_processor.load_iter(valid_X, valid_y, batch_size, False, num_workers)

    model = load_df_class()(num_classes)
    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=learning_rate)
    model.to(device)

    ckp_dir = checkpoint_dir(cfg, dataset)
    ckp_dir.mkdir(parents=True, exist_ok=True)
    out_file = ckp_dir / f"{save_name}.pth"
    if out_file.exists():
        print(f"Using existing checkpoint: {out_file}")
    else:
        criterion = torch.nn.CrossEntropyLoss()
        best_f1 = -1.0
        best_epoch = -1
        for epoch in progress(range(epochs), desc=f"{dataset} train epochs", unit="epoch"):
            model.train()
            loss_sum = 0.0
            count = 0
            train_batches = progress(train_iter, desc=f"{dataset} epoch {epoch} train", unit="batch", leave=False)
            for cur_X, cur_y in train_batches:
                cur_X = cur_X.to(device)
                cur_y = cur_y.to(device)
                optimizer.zero_grad()
                logits = model(cur_X)
                loss = criterion(logits, cur_y)
                loss.backward()
                optimizer.step()
                loss_sum += float(loss.detach().cpu().item()) * int(cur_y.numel())
                count += int(cur_y.numel())
                train_batches.set_postfix(loss=f"{float(loss.detach().cpu().item()):.4f}")

            model.eval()
            valid_true = []
            valid_pred = []
            with torch.no_grad():
                for cur_X, cur_y in progress(valid_iter, desc=f"{dataset} epoch {epoch} val", unit="batch", leave=False):
                    logits = model(cur_X.to(device))
                    pred = torch.argmax(logits, dim=1).cpu().numpy()
                    valid_pred.append(pred)
                    valid_true.append(cur_y.numpy())
            valid_true = np.concatenate(valid_true)
            valid_pred = np.concatenate(valid_pred)
            macro_f1 = float(f1_score(valid_true, valid_pred, average="macro"))
            train_loss = loss_sum / max(count, 1)
            print(f"epoch {epoch}: train_loss={train_loss:.6f}, valid_macro_f1={macro_f1:.6f}")
            if macro_f1 > best_f1:
                best_f1 = macro_f1
                best_epoch = epoch
                torch.save(model.state_dict(), str(out_file))
            print(f"best epoch {best_epoch}: macro_f1={best_f1:.6f}")

    train_record = {
        "dataset": dataset,
        "model": cfg.get("model", "DF"),
        "checkpoint": str(out_file),
        "device": device_name,
        "seed": seed,
        "seq_len": seq_len,
        "feature": cfg.get("feature", "DIR"),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "optimizer": optimizer_name,
    }
    write_json(ckp_dir / "train_config.json", train_record)
    return train_record


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train DF for fair comparison using generated PCAP-derived WFlib data.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    device = args.device or cfg.get("train", {}).get("device", "cuda")
    record = train(cfg, args.dataset, device)
    print(record)
    return 0


if __name__ == "__main__":
    sys.exit(main())
