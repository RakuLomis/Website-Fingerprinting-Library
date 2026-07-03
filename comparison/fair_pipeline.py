import argparse
import sys
from pathlib import Path

from comparison.common import checkpoint_dir, display_path, load_config, python_cmd, repo_root, run_cmd
from comparison.feature_builders import base_data_ready, missing_model_data, model_data_ready
from comparison.model_registry import model_config
from comparison.model_registry import default_models


def split_by_data_need(cfg, datasets, models):
    missing_base = [dataset for dataset in datasets if not base_data_ready(cfg, dataset)]
    missing_features = []
    ready_features = []
    for dataset in datasets:
        for model_name in models:
            mcfg = model_config(model_name)
            if not mcfg.get("derived_feature"):
                continue
            if model_data_ready(cfg, dataset, model_name):
                ready_features.append((dataset, model_name))
            else:
                missing_features.append((dataset, model_name))
    return missing_base, missing_features, ready_features


def require_model_data(cfg, datasets, models):
    missing = []
    for dataset in datasets:
        for model_name in models:
            missing.extend(missing_model_data(cfg, dataset, model_name))
    if missing:
        formatted = "\n".join(f"  - {display_path(path)}" for path in missing)
        raise FileNotFoundError(
            "Required model data files are missing. Run stages `preprocess features` first, "
            "or place existing generated data under the configured relative generated_root.\n"
            f"{formatted}"
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the WFlib multi-model fair-comparison pipeline.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--stages", nargs="+", default=["preprocess", "features", "train", "eval", "profile", "latency", "summary"],
                        choices=["preprocess", "features", "train", "eval", "profile", "latency", "summary"])
    parser.add_argument("--device", default=None, help="Training/eval device. Defaults to config train.device, usually cuda.")
    parser.add_argument("--overwrite-data", action="store_true", help="Regenerate base and derived feature files even when complete files already exist.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    datasets = args.datasets or cfg.get("datasets", [])
    models = args.models or cfg.get("models", default_models())
    py = python_cmd(cfg)
    root = repo_root()
    config_arg = ["--config", str(Path(args.config).resolve())] if args.config else []
    device = args.device or cfg.get("train", {}).get("device", "cuda")
    save_name = cfg.get("train", {}).get("save_name", "max_f1")

    missing_base, missing_features, ready_features = split_by_data_need(cfg, datasets, models)

    if "preprocess" in args.stages:
        if args.overwrite_data:
            run_cmd(py + ["-m", "comparison.df_pcap_to_npz"] + config_arg + ["--datasets"] + datasets + ["--overwrite"], args.dry_run, cwd=root)
        elif missing_base:
            run_cmd(py + ["-m", "comparison.df_pcap_to_npz"] + config_arg + ["--datasets"] + missing_base, args.dry_run, cwd=root)
        else:
            print("All requested base train/val/test data already exist; skip PCAP preprocessing.")

    if "features" in args.stages:
        if args.overwrite_data:
            run_cmd(py + ["-m", "comparison.feature_builders"] + config_arg + ["--datasets"] + datasets + ["--models"] + models + ["--overwrite"], args.dry_run, cwd=root)
        elif missing_features:
            feature_datasets = sorted({dataset for dataset, _ in missing_features})
            feature_models = sorted({model_name for _, model_name in missing_features})
            run_cmd(py + ["-m", "comparison.feature_builders"] + config_arg + ["--datasets"] + feature_datasets + ["--models"] + feature_models, args.dry_run, cwd=root)
        else:
            print("All requested model feature data already exist; skip derived feature building.")

    data_using_stages = {"train", "eval", "profile", "latency"}
    if data_using_stages.intersection(args.stages) and not args.dry_run:
        require_model_data(cfg, datasets, models)

    for dataset in datasets:
        for model_name in models:
            checkpoint = checkpoint_dir(cfg, dataset, model_name) / f"{save_name}.pth"
            checkpoint_arg = display_path(checkpoint)
            if "train" in args.stages:
                run_cmd(py + ["-m", "comparison.train_model"] + config_arg + ["--dataset", dataset, "--model", model_name, "--device", device], args.dry_run, cwd=root)
            if "eval" in args.stages:
                run_cmd(py + ["-m", "comparison.eval_metrics"] + config_arg + ["--dataset", dataset, "--model", model_name, "--checkpoint", checkpoint_arg, "--device", device], args.dry_run, cwd=root)
            if "profile" in args.stages:
                run_cmd(py + ["-m", "comparison.profile_model"] + config_arg + ["--dataset", dataset, "--model", model_name], args.dry_run, cwd=root)
            if "latency" in args.stages:
                run_cmd(py + ["-m", "comparison.cpu_benchmark"] + config_arg + ["--dataset", dataset, "--model", model_name, "--checkpoint", checkpoint_arg], args.dry_run, cwd=root)

    if "summary" in args.stages:
        run_cmd(py + ["-m", "comparison.summarize_all_results"] + config_arg + ["--datasets"] + datasets + ["--models"] + models, args.dry_run, cwd=root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
