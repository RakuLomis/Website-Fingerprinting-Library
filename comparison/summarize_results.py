import argparse
import csv
import sys
from pathlib import Path

from comparison.common import dataset_dir, load_config, read_json, write_json


SUMMARY_FIELDS = [
    "dataset",
    "model",
    "accuracy",
    "macro_f1",
    "loss",
    "total_params",
    "trainable_params",
    "active_params_estimated",
    "active_param_ratio",
    "dense_equivalent_flops",
    "effective_flops",
    "cpu_latency_p50_ms",
    "cpu_latency_p90_ms",
    "cpu_latency_p95_ms",
    "cpu_latency_p99_ms",
    "cpu_latency_mean_ms",
    "cpu_throughput_samples_per_s",
    "preprocess_policy",
    "flow_leakage_count",
]


def maybe_read(path):
    path = Path(path)
    return read_json(path) if path.exists() else {}


def summarize(cfg, datasets):
    rows = []
    results_root = Path(cfg["results_root"])
    model = cfg.get("model", "DF")
    for dataset in datasets:
        metrics = maybe_read(results_root / "metrics" / dataset / model / "_metrics.json")
        profile = maybe_read(results_root / "profiles" / dataset / model / "_profile.json")
        latency = maybe_read(results_root / "latency" / dataset / model / "_cpu_latency.json")
        preprocess = maybe_read(dataset_dir(cfg, dataset) / "preprocess_summary.json")
        row = {field: None for field in SUMMARY_FIELDS}
        row.update({
            "dataset": dataset,
            "model": model,
            "preprocess_policy": preprocess.get("preprocess_policy", cfg.get("preprocess_policy")),
            "flow_leakage_count": preprocess.get("flow_leakage_count"),
        })
        for source in [metrics, profile, latency]:
            for field in SUMMARY_FIELDS:
                if field in source:
                    row[field] = source[field]
        rows.append(row)

    write_json(results_root / "comparison_summary.json", rows)
    with (results_root / "comparison_summary.csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description="Summarize DF fair-comparison outputs.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--datasets", nargs="+", default=None)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    datasets = args.datasets or cfg.get("datasets", [])
    rows = summarize(cfg, datasets)
    for row in rows:
        print(row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
