import argparse
import sys
from pathlib import Path

import numpy as np

from WFlib.tools import data_processor
from comparison.common import dataset_dir, display_path, ensure_dir, load_config, progress
from comparison.model_registry import default_models, model_config


SPLITS = ["train", "val", "test"]
BASE_AUDIT_FILES = ["label_map.json", "split_manifest.tsv"]


def model_data_dir(cfg, dataset, model_name):
    mcfg = model_config(model_name)
    derived = mcfg.get("derived_feature")
    if not derived:
        return dataset_dir(cfg, dataset)
    return dataset_dir(cfg, dataset) / "features" / model_name


def base_dataset_dir(cfg, dataset):
    return dataset_dir(cfg, dataset)


def feature_file_path(cfg, dataset, model_name, split):
    return model_data_dir(cfg, dataset, model_name) / f"{split}.npz"


def base_data_ready(cfg, dataset):
    base_dir = base_dataset_dir(cfg, dataset)
    required = [base_dir / f"{split}.npz" for split in SPLITS]
    required.extend(base_dir / name for name in BASE_AUDIT_FILES)
    return all(path.exists() for path in required)


def model_data_ready(cfg, dataset, model_name):
    model_cfg = model_config(model_name)
    if not base_data_ready(cfg, dataset):
        return False
    if not model_cfg.get("derived_feature"):
        return True
    return all(feature_file_path(cfg, dataset, model_name, split).exists() for split in SPLITS)


def missing_model_data(cfg, dataset, model_name):
    missing = []
    base_dir = base_dataset_dir(cfg, dataset)
    for split in SPLITS:
        path = base_dir / f"{split}.npz"
        if not path.exists():
            missing.append(path)
    for name in BASE_AUDIT_FILES:
        path = base_dir / name
        if not path.exists():
            missing.append(path)
    model_cfg = model_config(model_name)
    if model_cfg.get("derived_feature"):
        for split in SPLITS:
            path = feature_file_path(cfg, dataset, model_name, split)
            if not path.exists():
                missing.append(path)
    return missing


def build_feature_array(feature_name, X, model_cfg):
    base_seq_len = int(model_cfg.get("base_seq_len", model_cfg.get("seq_len", 5000)))
    X = data_processor.length_align(X, base_seq_len)
    if feature_name == "TAM":
        return data_processor.extract_TAM(X)
    if feature_name == "MTAF":
        return data_processor.extract_MTAF(X)
    if feature_name == "TAF":
        return data_processor.extract_TAF(X)
    raise ValueError(f"Unsupported derived feature: {feature_name}")


def prepare_model_features(cfg, dataset, model_name, overwrite=False):
    model_cfg = model_config(model_name)
    derived = model_cfg.get("derived_feature")
    if not derived:
        if not base_data_ready(cfg, dataset):
            missing = ", ".join(str(path) for path in missing_model_data(cfg, dataset, model_name))
            raise FileNotFoundError(f"{dataset}/{model_name} requires existing base split files. Missing: {missing}")
        return {
            "dataset": dataset,
            "model": model_name,
            "feature": model_cfg["feature"],
            "derived": False,
            "data_dir": str(dataset_dir(cfg, dataset)),
        }

    out_dir = model_data_dir(cfg, dataset, model_name)
    ensure_dir(out_dir)
    built = {}
    for split in progress(SPLITS, desc=f"{dataset}/{model_name} build {derived}", unit="split"):
        out_file = out_dir / f"{split}.npz"
        if out_file.exists() and not overwrite:
            built[split] = "exists"
            continue
        data = np.load(dataset_dir(cfg, dataset) / f"{split}.npz")
        X = data["X"]
        y = data["y"]
        feat_X = build_feature_array(derived, X, model_cfg)
        np.savez_compressed(out_file, X=feat_X, y=y)
        built[split] = str(feat_X.shape)
    return {
        "dataset": dataset,
        "model": model_name,
        "feature": model_cfg["feature"],
        "derived": True,
        "derived_feature": derived,
        "data_dir": str(out_dir),
        "splits": built,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Prepare derived WFlib features for configured models.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--check-only", action="store_true", help="Only report whether every requested model has usable data files.")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    datasets = args.datasets or cfg.get("datasets", [])
    models = args.models or cfg.get("models", default_models())
    for dataset in datasets:
        for model_name in models:
            if args.check_only:
                ready = model_data_ready(cfg, dataset, model_name)
                info = {
                    "dataset": dataset,
                    "model": model_name,
                    "ready": ready,
                    "data_dir": display_path(model_data_dir(cfg, dataset, model_name)),
                    "missing": [display_path(path) for path in missing_model_data(cfg, dataset, model_name)],
                }
                print(info)
                continue
            info = prepare_model_features(cfg, dataset, model_name, args.overwrite)
            print(info)
    return 0


if __name__ == "__main__":
    sys.exit(main())
