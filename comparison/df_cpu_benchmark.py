import argparse
import os
import statistics
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import torch

from WFlib.tools import data_processor
from comparison.common import dataset_dir, load_config, load_df_class, progress, result_subdir, write_json


def percentile(values, pct):
    if not values:
        return None
    return float(np.percentile(np.array(values, dtype=np.float64), pct))


def rounded(value):
    return round(float(value), 6) if value is not None else None


def repeated_batches(X, total, batch_size):
    n = int(X.shape[0])
    if n == 0:
        return
    made = 0
    while made < total:
        take = min(batch_size, total - made)
        indices = [(made + i) % n for i in range(take)]
        yield X[indices]
        made += take


def benchmark(cfg, dataset, checkpoint):
    bench_cfg = cfg.get("cpu_benchmark", {})
    seq_len = int(cfg.get("seq_len", 5000))
    latency_batch_size = int(bench_cfg.get("latency_batch_size", 1))
    throughput_batch_size = int(bench_cfg.get("throughput_batch_size", 32))
    warmup_samples = int(bench_cfg.get("warmup_samples", 100))
    measured_samples = int(bench_cfg.get("measured_samples", 1000))
    num_threads = bench_cfg.get("num_threads")
    if num_threads is not None:
        torch.set_num_threads(int(num_threads))

    X, y = data_processor.load_data(str(dataset_dir(cfg, dataset) / "test.npz"), cfg.get("feature", "DIR"), seq_len, 1)
    num_classes = int(y.max().item()) + 1
    model = load_df_class()(num_classes)
    model.load_state_dict(torch.load(str(checkpoint), map_location="cpu"))
    model.cpu()
    model.eval()

    with torch.no_grad():
        warmup_bar = progress(repeated_batches(X, warmup_samples, latency_batch_size), total=warmup_samples, desc=f"{dataset} CPU warmup", unit="sample")
        for batch in warmup_bar:
            model(batch.cpu())
            warmup_bar.update(int(batch.shape[0]) - 1)

    latencies_ms = []
    with torch.no_grad():
        latency_bar = progress(repeated_batches(X, measured_samples, latency_batch_size), total=measured_samples, desc=f"{dataset} CPU latency", unit="sample")
        for batch in latency_bar:
            start = time.perf_counter()
            model(batch.cpu())
            elapsed = time.perf_counter() - start
            latencies_ms.append((elapsed / int(batch.shape[0])) * 1000.0)
            latency_bar.update(int(batch.shape[0]) - 1)

    throughput_seen = 0
    throughput_start = time.perf_counter()
    with torch.no_grad():
        throughput_bar = progress(repeated_batches(X, measured_samples, throughput_batch_size), total=measured_samples, desc=f"{dataset} CPU throughput", unit="sample")
        for batch in throughput_bar:
            model(batch.cpu())
            throughput_seen += int(batch.shape[0])
            throughput_bar.update(int(batch.shape[0]) - 1)
    throughput_elapsed = time.perf_counter() - throughput_start

    result = {
        "dataset": dataset,
        "model": cfg.get("model", "DF"),
        "device": "cpu",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "cpu_threads": int(torch.get_num_threads()),
        "latency_batch_size": latency_batch_size,
        "throughput_batch_size": throughput_batch_size,
        "seq_length": seq_len,
        "input_length": seq_len,
        "warmup_samples": warmup_samples,
        "measured_samples": measured_samples,
        "cpu_latency_p50_ms": rounded(percentile(latencies_ms, 50)),
        "cpu_latency_p90_ms": rounded(percentile(latencies_ms, 90)),
        "cpu_latency_p95_ms": rounded(percentile(latencies_ms, 95)),
        "cpu_latency_p99_ms": rounded(percentile(latencies_ms, 99)),
        "cpu_latency_mean_ms": round(float(statistics.mean(latencies_ms)), 6) if latencies_ms else None,
        "cpu_throughput_samples_per_s": round(float(throughput_seen / throughput_elapsed), 6) if throughput_elapsed > 0 else None,
        "checkpoint": str(checkpoint),
    }
    write_json(result_subdir(cfg, "latency", dataset) / "_cpu_latency.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Benchmark DF CPU latency and throughput.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    result = benchmark(cfg, args.dataset, Path(args.checkpoint))
    print(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
