import argparse
import csv
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from scapy.all import IP, IPv6, PcapReader, TCP, UDP
except ImportError:  # pragma: no cover - handled at runtime for user clarity
    IP = IPv6 = PcapReader = TCP = UDP = None

from comparison.common import DEFAULT_DATASETS, ensure_dir, load_config, progress


PCAP_SUFFIXES = {".pcap", ".pcapng", ".cap"}
REQUIRED_BASE_FILES = ["train.npz", "val.npz", "test.npz", "label_map.json", "split_manifest.tsv"]


def canonical_flow_key(src, dst, sport, dport, proto):
    left = (str(src), int(sport))
    right = (str(dst), int(dport))
    if right < left:
        left, right = right, left
    return f"{left[0]}:{left[1]}-{right[0]}:{right[1]}-{proto}"


def flow_digest(flow_key):
    return hashlib.sha256(flow_key.encode("utf-8")).hexdigest()[:24]


def label_from_path(dataset_root, pcap_path):
    rel = pcap_path.relative_to(dataset_root)
    if len(rel.parts) <= 1:
        return dataset_root.name
    return rel.parts[0]


def iter_pcaps(dataset_root):
    for path in sorted(dataset_root.rglob("*")):
        if path.is_file() and path.suffix.lower() in PCAP_SUFFIXES:
            yield path


def packet_tuple(pkt):
    if IP in pkt:
        ip = pkt[IP]
        proto = int(ip.proto)
        src, dst = ip.src, ip.dst
    elif IPv6 in pkt:
        ip = pkt[IPv6]
        proto = int(ip.nh)
        src, dst = ip.src, ip.dst
    else:
        return None

    if TCP in pkt:
        sport, dport = int(pkt[TCP].sport), int(pkt[TCP].dport)
        proto_name = "TCP"
    elif UDP in pkt:
        sport, dport = int(pkt[UDP].sport), int(pkt[UDP].dport)
        proto_name = "UDP"
    else:
        sport, dport = 0, 0
        proto_name = str(proto)
    return src, dst, sport, dport, proto_name


def extract_flow_samples(pcap_path, min_packets=2):
    flows = {}
    first_seen = {}
    with PcapReader(str(pcap_path)) as reader:
        for pkt in reader:
            fields = packet_tuple(pkt)
            if fields is None:
                continue
            src, dst, sport, dport, proto = fields
            key = canonical_flow_key(src, dst, sport, dport, proto)
            endpoint = (str(src), int(sport))
            if key not in flows:
                flows[key] = []
                first_seen[key] = endpoint
                t0 = float(pkt.time)
                flows[key].append([1.0, 1e-6, t0])
                continue
            direction = 1.0 if endpoint == first_seen[key] else -1.0
            t0 = flows[key][0][2]
            rel_time = max(float(pkt.time) - t0, 1e-6)
            flows[key].append([direction, rel_time, t0])

    samples = []
    for key, packets in flows.items():
        if len(packets) < min_packets:
            continue
        seq = np.array([direction * rel_time for direction, rel_time, _ in packets], dtype=np.float32)
        samples.append((key, seq))
    return samples


def assign_splits(flow_keys, ratios, seed):
    ordered = sorted(set(flow_keys))
    rng = random.Random(seed)
    rng.shuffle(ordered)
    n = len(ordered)
    n_train = int(n * ratios["train"])
    n_val = int(n * ratios["val"])
    split_by_flow = {}
    for idx, key in enumerate(ordered):
        if idx < n_train:
            split_by_flow[key] = "train"
        elif idx < n_train + n_val:
            split_by_flow[key] = "val"
        else:
            split_by_flow[key] = "test"
    return split_by_flow


def flow_leakage_count(rows):
    seen = defaultdict(set)
    for row in rows:
        seen[row["flow_key"]].add(row["split"])
    return sum(1 for splits in seen.values() if len(splits) > 1)


def base_dataset_exists(out_dir):
    return all((out_dir / name).exists() for name in REQUIRED_BASE_FILES)


def align_sequence(seq, seq_len):
    out = np.zeros(seq_len, dtype=np.float32)
    limit = min(seq_len, len(seq))
    if limit:
        out[:limit] = seq[:limit]
    return out


def write_npz(out_dir, split, samples, labels, label_to_id, seq_len):
    split_samples = samples.get(split, [])
    X = np.stack([align_sequence(seq, seq_len) for seq, _ in split_samples]).astype(np.float32) if split_samples else np.empty((0, seq_len), dtype=np.float32)
    if len(split_samples) == 0:
        X = np.empty((0, seq_len), dtype=np.float32)
    y = np.array([label_to_id[label] for _, label in split_samples], dtype=np.int64)
    np.savez_compressed(out_dir / f"{split}.npz", X=X, y=y)


def process_dataset(raw_root, out_root, dataset, ratios, seed, min_packets, seq_len, overwrite=False):
    if PcapReader is None:
        raise RuntimeError("scapy is required to parse PCAP files. Install it in Pytorch_env, e.g. `pip install scapy`.")

    dataset_root = raw_root / dataset
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_root}")

    out_dir = out_root / dataset
    ensure_dir(out_dir)
    if base_dataset_exists(out_dir) and not overwrite:
        summary_path = out_dir / "preprocess_summary.json"
        if summary_path.exists():
            with summary_path.open("r", encoding="utf-8") as fp:
                summary = json.load(fp)
        else:
            summary = {
                "dataset": dataset,
                "status": "existing",
                "data_dir": str(out_dir),
                "flow_leakage_count": None,
                "preprocess_policy": "Existing base WFlib split files reused; no PCAP parsing performed.",
            }
        summary["status"] = "existing_reused"
        summary["data_dir"] = str(out_dir)
        print(f"{dataset}: found existing base data at {out_dir}; skip PCAP preprocessing.")
        return summary
    skipped_path = out_dir / "skipped_files.log"
    skipped = []
    records = []

    pcap_files = list(iter_pcaps(dataset_root))
    for pcap in progress(pcap_files, desc=f"{dataset} parse pcaps", unit="pcap"):
        label = label_from_path(dataset_root, pcap)
        try:
            for flow_key, seq in extract_flow_samples(pcap, min_packets=min_packets):
                records.append({
                    "label": label,
                    "source_file": str(pcap.relative_to(raw_root)),
                    "flow_key": flow_digest(flow_key),
                    "sequence": seq,
                })
        except Exception as exc:  # keep batch conversion moving and auditable
            skipped.append(f"{pcap.relative_to(raw_root)}\t{type(exc).__name__}: {exc}")

    labels = sorted({row["label"] for row in records})
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    split_by_flow = assign_splits([row["flow_key"] for row in records], ratios, seed)
    samples = {"train": [], "val": [], "test": []}
    manifest_rows = []

    for row in progress(records, desc=f"{dataset} assign splits", unit="sample"):
        split = split_by_flow[row["flow_key"]]
        samples[split].append((row["sequence"], row["label"]))
        manifest_rows.append({
            "split": split,
            "label": row["label"],
            "source_file": row["source_file"],
            "flow_key": row["flow_key"],
        })

    for split in progress(["train", "val", "test"], desc=f"{dataset} write npz", unit="split"):
        write_npz(out_dir, split, samples, labels, label_to_id, seq_len)

    with (out_dir / "label_map.json").open("w", encoding="utf-8") as fp:
        json.dump(label_to_id, fp, indent=2, sort_keys=True)

    with (out_dir / "split_manifest.tsv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["split", "label", "source_file", "flow_key"], delimiter="\t")
        writer.writeheader()
        writer.writerows(manifest_rows)

    skipped_path.write_text("\n".join(skipped) + ("\n" if skipped else ""), encoding="utf-8")

    summary = {
        "dataset": dataset,
        "num_samples": len(records),
        "num_labels": len(labels),
        "split_counts": {split: len(samples[split]) for split in ["train", "val", "test"]},
        "seq_len": seq_len,
        "flow_leakage_count": flow_leakage_count(manifest_rows),
        "skipped_files": len(skipped),
        "preprocess_policy": "DF DIR features only; ETH/IP/port/protocol/SNI excluded from model inputs; hashed bidirectional five-tuple used only for split auditing.",
    }
    with (out_dir / "preprocess_summary.json").open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2, sort_keys=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Convert raw PCAP datasets into DF-compatible flow-safe WFlib NPZ files.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--raw-root", default=None)
    parser.add_argument("--out-root", default=None)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--min-packets", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true", help="Regenerate base train/val/test data even if complete split files already exist.")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    raw_root = Path(args.raw_root or cfg["raw_root"])
    out_root = Path(args.out_root or cfg["generated_root"])
    datasets = args.datasets or cfg.get("datasets", DEFAULT_DATASETS)
    seed = args.seed if args.seed is not None else int(cfg.get("seed", 42))
    seq_len = args.seq_len if args.seq_len is not None else int(cfg.get("base_seq_len", cfg.get("seq_len", 5000)))
    ratios = cfg.get("split_ratios", {"train": 0.8, "val": 0.1, "test": 0.1})

    all_summaries = []
    for dataset in datasets:
        summary = process_dataset(raw_root, out_root, dataset, ratios, seed, args.min_packets, seq_len, args.overwrite)
        print(json.dumps(summary, indent=2, sort_keys=True))
        all_summaries.append(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
