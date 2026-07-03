import argparse
import random
import sys

import numpy as np
import torch
from sklearn.metrics import f1_score

from WFlib.tools import data_processor
from comparison.common import checkpoint_dir, instantiate_model, load_config, progress, write_json
from comparison.feature_builders import model_data_dir
from comparison.model_registry import model_config


def train(cfg, dataset, model_name, device_name):
    model_cfg = model_config(model_name)
    if model_cfg.get("loss") != "CrossEntropyLoss":
        raise NotImplementedError(f"{model_name} uses {model_cfg.get('loss')} and is not enabled in the generic supervised trainer.")

    seed = int(cfg.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_cfg = cfg.get("train", {})
    seq_len = int(model_cfg["seq_len"])
    batch_size = int(model_cfg.get("train_batch_size", 128))
    num_workers = int(train_cfg.get("num_workers", 10))
    epochs = int(model_cfg.get("train_epochs", 30))
    learning_rate = float(model_cfg.get("learning_rate", 2e-3))
    optimizer_name = model_cfg.get("optimizer", "Adam")
    save_name = train_cfg.get("save_name", "max_f1")

    if device_name.startswith("cuda"):
        assert torch.cuda.is_available(), f"The specified device {device_name} does not exist"
    device = torch.device(device_name)

    data_dir = model_data_dir(cfg, dataset, model_name)
    train_X, train_y = data_processor.load_data(str(data_dir / "train.npz"), model_cfg["feature"], seq_len, 1)
    valid_X, valid_y = data_processor.load_data(str(data_dir / "val.npz"), model_cfg["feature"], seq_len, 1)
    num_classes = int(train_y.max().item()) + 1

    train_iter = data_processor.load_iter(train_X, train_y, batch_size, True, num_workers)
    valid_iter = data_processor.load_iter(valid_X, valid_y, batch_size, False, num_workers)

    model = instantiate_model(model_cfg, num_classes, num_tabs=1)
    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=learning_rate)
    scheduler = None
    if model_cfg.get("lradj") == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.74)
    model.to(device)

    ckp_dir = checkpoint_dir(cfg, dataset, model_name)
    ckp_dir.mkdir(parents=True, exist_ok=True)
    out_file = ckp_dir / f"{save_name}.pth"
    if out_file.exists():
        print(f"Using existing checkpoint: {out_file}")
    else:
        criterion = torch.nn.CrossEntropyLoss()
        best_f1 = -1.0
        best_epoch = -1
        for epoch in progress(range(epochs), desc=f"{dataset}/{model_name} train epochs", unit="epoch"):
            model.train()
            loss_sum = 0.0
            count = 0
            train_batches = progress(train_iter, desc=f"{dataset}/{model_name} epoch {epoch} train", unit="batch", leave=False)
            for cur_X, cur_y in train_batches:
                cur_X = cur_X.to(device)
                cur_y = cur_y.to(device)
                optimizer.zero_grad()
                logits = model(cur_X)
                loss = criterion(logits, cur_y)
                loss.backward()
                optimizer.step()
                loss_value = float(loss.detach().cpu().item())
                loss_sum += loss_value * int(cur_y.numel())
                count += int(cur_y.numel())
                train_batches.set_postfix(loss=f"{loss_value:.4f}")

            model.eval()
            valid_true = []
            valid_pred = []
            with torch.no_grad():
                for cur_X, cur_y in progress(valid_iter, desc=f"{dataset}/{model_name} epoch {epoch} val", unit="batch", leave=False):
                    logits = model(cur_X.to(device))
                    valid_pred.append(torch.argmax(logits, dim=1).cpu().numpy())
                    valid_true.append(cur_y.numpy())
            valid_true = np.concatenate(valid_true)
            valid_pred = np.concatenate(valid_pred)
            macro_f1 = float(f1_score(valid_true, valid_pred, average="macro"))
            train_loss = loss_sum / max(count, 1)
            print(f"{dataset}/{model_name} epoch {epoch}: train_loss={train_loss:.6f}, valid_macro_f1={macro_f1:.6f}")
            if macro_f1 > best_f1:
                best_f1 = macro_f1
                best_epoch = epoch
                torch.save(model.state_dict(), str(out_file))
            print(f"{dataset}/{model_name} best epoch {best_epoch}: macro_f1={best_f1:.6f}")
            if scheduler is not None:
                scheduler.step()

    train_record = {
        "dataset": dataset,
        "model": model_name,
        "checkpoint": str(out_file),
        "device": device_name,
        "seed": seed,
        "seq_len": seq_len,
        "feature": model_cfg["feature"],
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "optimizer": optimizer_name,
    }
    write_json(ckp_dir / "train_config.json", train_record)
    return train_record


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train one WFlib model for fair comparison.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    device = args.device or cfg.get("train", {}).get("device", "cuda")
    print(train(cfg, args.dataset, args.model, device))
    return 0


if __name__ == "__main__":
    sys.exit(main())
