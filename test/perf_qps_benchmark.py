#!/usr/bin/env python
"""
Drive MemoryManagement against different backends with dataset samples at a target QPS.

Features:
  - Can rebuild an index from a Faiss IVF checkpoint first (--faiss-path), then drive requests.
  - Uses dataset items as independent requests (flattened from dataset/<name>.py).
  - Supports three access patterns:
        * per_item_rw:       every item does a search then an insert
        * read_all_write_tail: every item searches, only the last item inserts
        * head_read_tail_write: first item searches, every item inserts
  - Rate limits by comparing current time to the start time (simple QPS gate).
  - Records per-query latency for insert and search.

Example:
    python test/perf_qps_benchmark.py \\
        --backend m3 --dataset agentgym --limit 200 \\
        --faiss-path /path/to/index.faiss \\
        --qps 50 100 --mode per_item_rw --top-k 3
"""

from __future__ import annotations

import argparse
import os
import math
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence
import json
from pathlib import Path

# Ensure repo-root imports work when script is executed from test/.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from AgentMemory.interface import MemoryManagement
from AgentMemory.types import MemoryItem, Metric
from dataset import (
    AgentGymDataset,
    GSM8KReasoningDataset,
    PRMStepwiseDataset,
    UltraChatDataset,
)


DATASET_LOADERS = {
    "agentgym": AgentGymDataset,
    "prm800k": PRMStepwiseDataset,
    "ultrachat": UltraChatDataset,
    "gsm8k": GSM8KReasoningDataset,
}

MODES = {
    "per_item_rw": "Per-item search then insert.",
    "read_all_write_tail": "Search every item; only the last item inserts.",
    "head_read_tail_write": "First item searches; every item inserts.",
}


@dataclass
class OpLatency:
    kind: str
    value: float
    item_idx: int
    qps: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QPS-driven dataset perf benchmark.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional YAML/JSON config file to override flags.",
    )
    parser.add_argument(
        "--backend",
        choices=["placeholder", "quake", "m3", "m3multi"],
        default="placeholder",
        help="Memory backend to use.",
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASET_LOADERS.keys()),
        default="agentgym",
        help="Dataset loader defined in dataset/.",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Dataset split (falls back to each loader default when omitted).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Maximum number of dataset items to turn into requests (<=0 means no limit).",
    )
    parser.add_argument(
        "--prefill",
        type=int,
        default=0,
        help="Optional number of leading items to insert once before timed runs.",
    )
    parser.add_argument(
        "--qps",
        type=float,
        nargs="+",
        default=[50.0],
        help="One or more QPS targets to sweep (sequential runs).",
    )
    parser.add_argument(
        "--mode",
        choices=list(MODES.keys()),
        default="per_item_rw",
        help="Read/write pattern to apply.",
    )
    parser.add_argument("--top-k", type=int, default=3, help="k for searches.")
    parser.add_argument("--nprobe", type=int, default=32, help="nprobe for searches.")
    parser.add_argument(
        "--l0-threshold",
        type=float,
        default=None,
        help="L0 new-cluster threshold (for backend=m3multi; default=inf).",
    )
    parser.add_argument(
        "--search-threshold",
        type=float,
        default=None,
        help="Early-stop search threshold (for backend=m3multi; default=inf).",
    )
    parser.add_argument(
        "--model-name",
        default="intfloat/e5-large-v2",
        help="Encoder model passed to MemoryManagement.",
    )
    parser.add_argument("--precision", default="fp32", help="Encoder precision.")
    parser.add_argument("--device", default=None, help="Optional torch device override.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Encoder batch size passed to MemoryManagement.",
    )
    parser.add_argument(
        "--index-handle",
        default="perf-demo",
        help="Base index handle; per-QPS runs append the QPS value.",
    )
    parser.add_argument(
        "--faiss-path",
        type=str,
        default=None,
        help="Optional Faiss IVF index to rebuild before benchmarking (uses backend rebuild_index_from_faiss).",
    )
    parser.add_argument(
        "--faiss-normalized",
        action="store_true",
        help="Pass normalized=True to rebuild_index_from_faiss (default False).",
    )
    return parser.parse_args()


def load_config(path: str | None) -> Dict[str, Any]:
    if not path:
        return {}
    p = Path(path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f"Config file not found: {p}")
    text = p.read_text(encoding="utf-8")
    try:
        try:
            import yaml  # type: ignore
        except ImportError:
            yaml = None
        if yaml is not None:
            data = yaml.safe_load(text)
        else:
            data = json.loads(text)
    except Exception as exc:
        raise RuntimeError(f"Failed to parse config file {p}: {exc}") from exc
    return data or {}


def flatten_dataset(dataset_name: str, split: str | None, limit: int | None) -> List[str]:
    dataset_cls = DATASET_LOADERS[dataset_name]
    ds_kwargs: Dict[str, object] = {}
    if limit is not None and limit > 0:
        ds_kwargs["limit"] = limit
    if split:
        ds_kwargs["split"] = split
    loader = dataset_cls(**ds_kwargs)
    texts: List[str] = []
    for agent in loader.get_data():
        for request in agent.get("requests", []):
            for entry in request.get("trace", []):
                text = _extract_text(entry)
                if not text:
                    continue
                texts.append(text)
                if limit is not None and len(texts) >= limit:
                    return texts
    return texts


def _extract_text(entry: Mapping[str, Any]) -> str | None:
    """
    Prefer the "text" field; fall back to common role fields.
    """
    if not isinstance(entry, Mapping):
        return None
    text = entry.get("text")
    if isinstance(text, str) and text.strip():
        return text.strip()
    candidates: List[str] = []
    for key in ("user", "assistant", "human", "gpt", "question", "answer"):
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            candidates.append(value.strip())
    if candidates:
        return " ".join(candidates)
    return None


class QPSLimiter:
    """
    Simple gate that allows at most qps operations since start_time.
    """

    def __init__(self, qps: float):
        if qps <= 0:
            raise ValueError("qps must be positive")
        self.qps = float(qps)
        self.start = time.perf_counter()
        self.processed = 0

    def wait_for_slot(self) -> None:
        while True:
            elapsed = time.perf_counter() - self.start
            allowed = math.floor(elapsed * self.qps)
            if allowed >= self.processed + 1:
                break
            time.sleep(0.001)
        self.processed += 1


def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * pct / 100.0
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return ordered[int(pos)]
    weight = pos - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize(latencies: List[OpLatency], kind: str) -> Dict[str, float]:
    values = [rec.value for rec in latencies if rec.kind == kind]
    if not values:
        return {
            "count": 0,
            "mean_ms": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
            "max_ms": 0.0,
        }
    return {
        "count": len(values),
        "mean_ms": statistics.fmean(values) * 1000.0,
        "p50_ms": percentile(values, 50) * 1000.0,
        "p95_ms": percentile(values, 95) * 1000.0,
        "p99_ms": percentile(values, 99) * 1000.0,
        "max_ms": max(values) * 1000.0,
    }


def run_insert(mm: MemoryManagement, index: str, text: str, item_idx: int, qps: float) -> OpLatency:
    mm.add_insert(index, [MemoryItem(id=str(item_idx), data=text)])
    start = time.perf_counter()
    mm.run()
    return OpLatency(kind="insert", value=time.perf_counter() - start, item_idx=item_idx, qps=qps)


def run_search(
    mm: MemoryManagement,
    index: str,
    text: str,
    top_k: int,
    nprobe: int,
    item_idx: int,
    qps: float,
) -> OpLatency:
    req_id = f"q-{item_idx}"
    mm.add_search(index, [MemoryItem(id=None, data=text)], k=top_k, request_id=req_id, nprobe=nprobe)
    start = time.perf_counter()
    mm.run()
    return OpLatency(kind="search", value=time.perf_counter() - start, item_idx=item_idx, qps=qps)


def prefill_index(
    mm: MemoryManagement,
    index: str,
    texts: Sequence[str],
) -> None:
    if not texts:
        return
    mm.add_insert(
        index,
        [MemoryItem(id=str(i), data=text) for i, text in enumerate(texts)],
    )
    mm.run()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    def cfg_get(key, default=None):
        return cfg.get(key, default) if isinstance(cfg, dict) else default

    # override select args from config if provided
    args.backend = cfg_get("backend", args.backend)
    args.top_k = int(cfg_get("top_k", args.top_k))
    args.nprobe = int(cfg_get("nprobe", args.nprobe))
    args.faiss_path = cfg_get("faiss_path", args.faiss_path)
    if "faiss_normalized" in cfg:
        args.faiss_normalized = bool(cfg["faiss_normalized"])
    if "batch_size" in cfg:
        args.batch_size = int(cfg["batch_size"])
    if "model_name" in cfg:
        args.model_name = str(cfg["model_name"])
    if "device" in cfg:
        args.device = cfg["device"]
    l0_threshold = cfg_get("l0_new_cluster_threshold", args.l0_threshold)
    search_threshold = cfg_get("search_threshold", args.search_threshold)
    l0_merge_threshold = cfg_get("l0_merge_threshold", None)
    l0_max_nlist = cfg_get("l0_max_nlist", None)
    l0_nlist = cfg_get("l0_nlist", None)
    l1_nlist = cfg_get("l1_nlist", None)
    l2_nlist = cfg_get("l2_nlist", None)
    if args.mode not in MODES:
        raise ValueError(f"Unknown mode {args.mode}")
    effective_limit = None if args.limit is not None and args.limit <= 0 else args.limit
    raw_texts = flatten_dataset(args.dataset, args.split, effective_limit)
    if not raw_texts:
        raise SystemExit("No dataset items found.")
    print(f"dataset items prepared: {len(raw_texts)}")

    mm = MemoryManagement(
        backend=args.backend,
        model_name=args.model_name,
        precision=args.precision,
        device=args.device,
        batch_size=args.batch_size,
    )

    for qps in args.qps:
        index_handle = f"{args.index_handle}-{qps}"
        params = {}
        if l0_threshold is not None:
            params["l0_new_cluster_threshold"] = l0_threshold
        if search_threshold is not None:
            params["search_threshold"] = search_threshold
        if l0_merge_threshold is not None:
            params["l0_merge_threshold"] = l0_merge_threshold
        if l0_max_nlist is not None:
            params["l0_max_nlist"] = l0_max_nlist
        if l0_nlist is not None:
            params["l0_nlist"] = l0_nlist
        if l1_nlist is not None:
            params["l1_nlist"] = l1_nlist
        if l2_nlist is not None:
            params["l2_nlist"] = l2_nlist
        mm.create_index(index_handle, metric=Metric.L2, params=params or None)
        if args.faiss_path:
            faiss_path = os.path.abspath(os.path.expanduser(args.faiss_path))
            if not os.path.isfile(faiss_path):
                raise FileNotFoundError(f"Faiss file not found: {faiss_path}")
            mm.rebuild_index_from_faiss(index_handle, path=faiss_path, normalized=args.faiss_normalized)

        prefill = max(0, min(args.prefill, len(raw_texts)))
        working_items = raw_texts
        if prefill:
            prefill_texts = raw_texts[:prefill]
            working_items = raw_texts[prefill:]
            prefill_index(mm, index_handle, prefill_texts)

        limiter = QPSLimiter(qps)
        records: List[OpLatency] = []
        total = len(working_items)
        for i, text in enumerate(working_items):
            limiter.wait_for_slot()
            if args.mode == "per_item_rw":
                records.append(run_search(mm, index_handle, text, args.top_k, args.nprobe, i, qps))
                records.append(run_insert(mm, index_handle, text, i, qps))
            elif args.mode == "read_all_write_tail":
                records.append(run_search(mm, index_handle, text, args.top_k, args.nprobe, i, qps))
                if i == total - 1:
                    records.append(run_insert(mm, index_handle, text, i, qps))
            elif args.mode == "head_read_tail_write":
                if i == 0:
                    records.append(run_search(mm, index_handle, text, args.top_k, args.nprobe, i, qps))
                records.append(run_insert(mm, index_handle, text, i, qps))
        search_stats = summarize(records, "search")
        insert_stats = summarize(records, "insert")
        print(f"QPS={qps} mode={args.mode} | "
              f"search(count={search_stats['count']}, "
              f"mean={search_stats['mean_ms']:.2f}ms, "
              f"p50={search_stats['p50_ms']:.2f}ms, "
              f"p95={search_stats['p95_ms']:.2f}ms, "
              f"p99={search_stats['p99_ms']:.2f}ms, "
              f"max={search_stats['max_ms']:.2f}ms) | "
              f"insert(count={insert_stats['count']}, "
              f"mean={insert_stats['mean_ms']:.2f}ms, "
              f"p50={insert_stats['p50_ms']:.2f}ms, "
              f"p95={insert_stats['p95_ms']:.2f}ms, "
              f"p99={insert_stats['p99_ms']:.2f}ms, "
              f"max={insert_stats['max_ms']:.2f}ms)")

    if hasattr(mm.backend, "close"):
        try:
            mm.backend.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
