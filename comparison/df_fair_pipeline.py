import argparse
import sys
from pathlib import Path

from comparison.common import checkpoint_dir, load_config, python_cmd, repo_root, run_cmd


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the DF fair-comparison pipeline end to end.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--stages", nargs="+", default=["preprocess", "train", "eval", "profile", "latency", "summary"],
                        choices=["preprocess", "train", "eval", "profile", "latency", "summary"])
    parser.add_argument("--device", default=None, help="Training/eval device, e.g. cuda, cuda:0, or cpu.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    datasets = args.datasets or cfg.get("datasets", [])
    py = python_cmd(cfg)
    root = repo_root()
    config_arg = ["--config", str(Path(args.config).resolve())] if args.config else []
    device = args.device or cfg.get("train", {}).get("device", "cuda")
    save_name = cfg.get("train", {}).get("save_name", "max_f1")

    if "preprocess" in args.stages:
        run_cmd(py + ["-m", "comparison.df_pcap_to_npz"] + config_arg + ["--datasets"] + datasets, args.dry_run, cwd=root)

    for dataset in datasets:
        checkpoint = checkpoint_dir(cfg, dataset) / f"{save_name}.pth"
        if "train" in args.stages:
            run_cmd(py + ["-m", "comparison.df_train"] + config_arg + ["--dataset", dataset, "--device", device], args.dry_run, cwd=root)
        if "eval" in args.stages:
            run_cmd(py + ["-m", "comparison.df_eval_metrics"] + config_arg + ["--dataset", dataset, "--checkpoint", str(checkpoint), "--device", device], args.dry_run, cwd=root)
        if "profile" in args.stages:
            run_cmd(py + ["-m", "comparison.df_profile"] + config_arg + ["--dataset", dataset], args.dry_run, cwd=root)
        if "latency" in args.stages:
            run_cmd(py + ["-m", "comparison.df_cpu_benchmark"] + config_arg + ["--dataset", dataset, "--checkpoint", str(checkpoint)], args.dry_run, cwd=root)

    if "summary" in args.stages:
        run_cmd(py + ["-m", "comparison.summarize_results"] + config_arg + ["--datasets"] + datasets, args.dry_run, cwd=root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
