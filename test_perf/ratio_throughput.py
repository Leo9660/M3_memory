#!/usr/bin/env python
"""
Perf harness that sweeps multiple request patterns over all datasets.

Modes:
- item_search_insert: each item does search then insert (1:1).
- step_search_then_update: search every item, then insert them once at the end.
- head_search_tail_insert: first item search-only, remaining items insert only.
- search_only: search every item, never insert.
- ratio (legacy): synthetic vectors with a search:insert ratio like "1:1".

Supports:
- Backend selection via --backend.
- Optional Faiss bootstrap via --faiss-index before timed runs.
- Dataset loading (all six under dataset/) with HF encoder, or synthetic vectors
  with a passthrough encoder for the ratio mode.
"""

from __future__ import annotations

import argparse
import math
import csv
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import numpy as np

# Allow `import AgentMemory` when executed from test_perf/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from AgentMemory.interface import MemoryManagement
from AgentMemory.types import MemoryItem, Metric
from dataset import (
    AgentGymDataset,
    GSM8KReasoningDataset,
    PRMStepwiseDataset,
    UltraChatDataset,
    UltraFeedbackDataset,
    XLAMFunctionCallingDataset,
)

DATASET_LOADERS = {
    "agentgym": AgentGymDataset,
    "gsm8k": GSM8KReasoningDataset,
    "prm800k": PRMStepwiseDataset,
    "ultrachat": UltraChatDataset,
    "ultrafeedback": UltraFeedbackDataset,
    "xlam_function_calling": XLAMFunctionCallingDataset,
}


class VectorPassthroughEncoder:
    """
    Minimal encoder that expects MemoryItem.data to already be a 1D vector.
    Keeps perf runs focused on the backend instead of HF model latency.
    """

    def __init__(self, dim: int, normalize: bool = True) -> None:
        self.dim = int(dim)
        self.normalize = bool(normalize)

    def _encode(self, items: List[MemoryItem]) -> np.ndarray:
        mat = np.vstack([self._to_vec(it.data) for it in items]).astype(np.float32, copy=False)
        if self.normalize and mat.size:
            norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8
            mat = mat / norms
        return mat

    def _to_vec(self, value: object) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if arr.shape[0] != self.dim:
            raise ValueError(f"Expected vector dim={self.dim}, got {arr.shape[0]}")
        return arr

    def encode_items(self, items: List[MemoryItem]) -> np.ndarray:
        return self._encode(items)

    def encode_queries(self, items: List[MemoryItem]) -> np.ndarray:
        return self._encode(items)


def parse_ratio(text: str) -> Tuple[float, float]:
    """Parse strings like '1:1' or '2/1' into (search, insert)."""
    cleaned = text.replace("/", ":")
    parts = cleaned.split(":")
    if len(parts) != 2:
        raise ValueError(f"Ratio must be formatted as search:insert, got {text!r}")
    try:
        search_r = float(parts[0])
        insert_r = float(parts[1])
    except ValueError as exc:
        raise ValueError(f"Invalid ratio numbers: {text}") from exc
    if search_r < 0 or insert_r <= 0:
        raise ValueError("Ratio values must be non-negative and insert > 0")
    return search_r, insert_r


def chunk(items: List[MemoryItem], size: int) -> Iterable[List[MemoryItem]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def flush_queue(mm: MemoryManagement, stats: Dict[str, float], run_idx: int) -> None:
    if not mm.queue:
        return
    start = time.perf_counter()
    result = mm.run()
    elapsed = time.perf_counter() - start
    searched = sum(len(v) for v in result.searches.values())
    inserted = result.upserted + result.updated

    stats["runs"] += 1
    stats["time"] += elapsed
    stats["inserted"] += inserted
    stats["searched"] += searched

    total_ops = inserted + searched
    throughput = total_ops / elapsed if elapsed > 0 else float("inf")
    print(
        f"[run {run_idx:03d}] +insert {inserted}, +search_q {searched}, "
        f"{elapsed*1000:.1f} ms ({throughput:.1f} ops/s)"
    )


def metric_from_str(name: str) -> Metric:
    key = name.strip().lower()
    if key in ("cos", "cosine"):
        return Metric.COSINE
    if key in ("ip", "inner", "dot"):
        return Metric.IP
    if key in ("l2", "euclidean"):
        return Metric.L2
    raise ValueError(f"Unsupported metric: {name}")


def build_vectors(count: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal((count, dim), dtype=np.float32)


def prepare_items(prefix: str, mat: np.ndarray) -> List[MemoryItem]:
    return [MemoryItem(id=f"{prefix}-{i}", data=row) for i, row in enumerate(mat)]


def _extract_text(entry: Mapping[str, Any]) -> str | None:
    if not isinstance(entry, Mapping):
        return None
    text = entry.get("text")
    if isinstance(text, str) and text.strip():
        return text.strip()
    for key in ("user", "assistant", "human", "gpt", "question", "answer"):
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def flatten_dataset(dataset_name: str, split: str | None, limit: int | None) -> List[str]:
    loader_cls = DATASET_LOADERS[dataset_name]
    kwargs: Dict[str, object] = {}
    if split:
        kwargs["split"] = split
    if limit and limit > 0:
        kwargs["limit"] = limit
    loader = loader_cls(**kwargs)
    texts: List[str] = []
    for agent in loader.get_data():
        for req in agent.get("requests", []):
            for entry in req.get("trace", []):
                text = _extract_text(entry)
                if not text:
                    continue
                texts.append(text)
                if limit and limit > 0 and len(texts) >= limit:
                    return texts
    return texts


def to_items(prefix: str, texts: List[str]) -> List[MemoryItem]:
    return [MemoryItem(id=f"{prefix}-{i}", data=txt) for i, txt in enumerate(texts)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Search/insert ratio perf benchmark.")
    parser.add_argument("--backend", choices=["placeholder", "quake", "m3", "m3multi"], default="m3")
    parser.add_argument("--index", default="perf-ratio", help="Index handle passed to MemoryManagement.")
    parser.add_argument("--mode", choices=["item_search_insert", "step_search_then_update", "head_search_tail_insert", "search_only", "ratio"], default="item_search_insert", help="Request scheduling pattern.")
    parser.add_argument("--dataset", choices=list(DATASET_LOADERS.keys()), default=None, help="Dataset name; when omitted, synthetic vectors are used (ratio mode).")
    parser.add_argument("--split", default=None, help="Dataset split override.")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of dataset items (<=0 means all).")
    parser.add_argument("--dim", type=int, default=1024, help="Vector dimension for synthetic data (ratio mode).")
    parser.add_argument("--metric", default="l2", help="Metric for create_index (cosine/ip/l2).")
    parser.add_argument("--insert-count", type=int, default=50000, help="Number of vectors to insert (ratio mode only).")
    parser.add_argument("--ratio", default="1:1", help="Search:insert ratio for mode=ratio, e.g., 1:1 or 2:1.")
    parser.add_argument("--insert-batch", type=int, default=512, help="Vectors per insert request.")
    parser.add_argument("--search-batch", type=int, default=256, help="Queries per search request.")
    parser.add_argument("--ops-per-run", type=int, default=128, help="Flush queue to backend after this many enqueued ops (0 = single run).")
    parser.add_argument("--top-k", type=int, default=10, help="k for searches.")
    parser.add_argument("--nprobe", type=int, default=32, help="nprobe for searches.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for reproducibility.")
    parser.add_argument("--faiss-index", type=str, default=None, help="Optional Faiss IVF/flat index to rebuild before the run.")
    parser.add_argument("--faiss-normalized", action="store_true", help="Pass normalized=True to rebuild_index_from_faiss.")
    parser.add_argument("--normalize", dest="normalize", action="store_true", help="L2-normalize vectors before sending to backend (default for synthetic).")
    parser.add_argument("--no-normalize", dest="normalize", action="store_false", help="Disable L2 normalization before sending to backend.")
    parser.add_argument("--encoder", choices=["transformer", "passthrough"], default="transformer", help="Encoder to use (transformer required for text datasets; passthrough only for synthetic).")
    parser.add_argument("--log-file", type=str, default=None, help="Optional path to append a TSV log row (mode, dataset, backend, throughput).")
    parser.set_defaults(normalize=True)
    args = parser.parse_args()

    metric = metric_from_str(args.metric)

    if args.dataset and args.encoder == "passthrough":
        raise ValueError("Passthrough encoder only supports synthetic vectors. Use --encoder transformer for datasets.")
    if args.dataset and args.mode == "ratio":
        raise ValueError("mode=ratio is only supported for synthetic runs (no dataset).")

    if args.dataset:
        split = args.split  # None -> dataset default split
        limit = args.limit if args.limit and args.limit > 0 else None
        texts = flatten_dataset(args.dataset, split, limit)
        if not texts:
            raise RuntimeError("No dataset texts were loaded.")
        items = to_items(args.dataset, texts)
        encoder = None  # use default TransformerEncoder from MemoryManagement
        mm = MemoryManagement(backend=args.backend, encoder=encoder, default_nprobe=args.nprobe)
    else:
        search_ratio, insert_ratio = parse_ratio(args.ratio)
        rng = np.random.default_rng(args.seed)
        encoder = VectorPassthroughEncoder(dim=args.dim, normalize=args.normalize)
        mm = MemoryManagement(backend=args.backend, encoder=encoder, default_nprobe=args.nprobe)
        insert_mat = build_vectors(args.insert_count, args.dim, rng)
        expected_search = int(math.ceil(args.insert_count * (search_ratio / insert_ratio)))
        search_mat = build_vectors(expected_search, args.dim, rng)
        items = prepare_items("ins", insert_mat)
        queries = prepare_items("q", search_mat)
        ratio_state = (search_ratio, insert_ratio, queries)

    index_id = mm.create_index(args.index, metric=metric)

    if args.faiss_index:
        faiss_path = Path(args.faiss_index).expanduser()
        if not faiss_path.is_file():
            raise FileNotFoundError(f"Faiss index not found: {faiss_path}")
        print(f"[init] rebuilding index from Faiss: {faiss_path}")
        mm.rebuild_index_from_faiss(index_id, path=str(faiss_path), normalized=args.faiss_normalized)

    queued_ops = 0
    run_idx = 1
    stats: Dict[str, float] = {"runs": 0, "time": 0.0, "inserted": 0.0, "searched": 0.0}

    def maybe_flush() -> None:
        nonlocal queued_ops, run_idx
        if args.ops_per_run > 0 and queued_ops >= args.ops_per_run:
            flush_queue(mm, stats, run_idx)
            run_idx += 1
            queued_ops = 0

    print(f"[config] backend={args.backend}, mode={args.mode}, dataset={args.dataset or 'synthetic'}")

    if args.mode == "ratio":
        search_ratio, insert_ratio, queries = ratio_state
        insert_batches = list(chunk(items, args.insert_batch))
        search_batches = list(chunk(queries, args.search_batch))
        dispatched_insert = 0
        dispatched_search = 0
        insert_idx = 0
        search_idx = 0

        while insert_idx < len(insert_batches) or search_idx < len(search_batches):
            target_search = dispatched_insert * (search_ratio / insert_ratio)
            do_search = (
                search_idx < len(search_batches)
                and (dispatched_search < target_search or insert_idx >= len(insert_batches))
            )

            if do_search:
                batch = search_batches[search_idx]
                mm.add_search(index_id, batch, args.top_k, nprobe=args.nprobe, request_id=f"search-{search_idx}")
                dispatched_search += len(batch)
                search_idx += 1
            elif insert_idx < len(insert_batches):
                batch = insert_batches[insert_idx]
                mm.add_insert(index_id, batch)
                dispatched_insert += len(batch)
                insert_idx += 1
            else:
                break

            queued_ops += 1
            maybe_flush()

    else:
        search_batches = list(chunk(to_items("q", [it.data for it in items]), args.search_batch))
        insert_batches = list(chunk(items, args.insert_batch))

        if args.mode == "item_search_insert":
            for sb in search_batches:
                mm.add_search(index_id, sb, args.top_k, nprobe=args.nprobe, request_id=f"search-{run_idx}-{queued_ops}")
                queued_ops += 1
                maybe_flush()
                # align insert batch to same size if available
                if insert_batches:
                    ib = insert_batches.pop(0)
                    mm.add_insert(index_id, ib)
                    queued_ops += 1
                    maybe_flush()

            # insert any remaining batches
            for ib in insert_batches:
                mm.add_insert(index_id, ib)
                queued_ops += 1
                maybe_flush()

        elif args.mode == "step_search_then_update":
            for sb in search_batches:
                mm.add_search(index_id, sb, args.top_k, nprobe=args.nprobe, request_id=f"search-{run_idx}-{queued_ops}")
                queued_ops += 1
                maybe_flush()
            for ib in insert_batches:
                mm.add_insert(index_id, ib)
                queued_ops += 1
                maybe_flush()

        elif args.mode == "head_search_tail_insert":
            if search_batches:
                first = search_batches[0]
                mm.add_search(index_id, first, args.top_k, nprobe=args.nprobe, request_id="search-head")
                queued_ops += 1
                maybe_flush()
            for ib in insert_batches:
                mm.add_insert(index_id, ib)
                queued_ops += 1
                maybe_flush()

        elif args.mode == "search_only":
            for sb in search_batches:
                mm.add_search(index_id, sb, args.top_k, nprobe=args.nprobe, request_id=f"search-{run_idx}-{queued_ops}")
                queued_ops += 1
                maybe_flush()
        else:
            raise ValueError(f"Unsupported mode: {args.mode}")

    if mm.queue:
        flush_queue(mm, stats, run_idx)

    total_ops = stats["inserted"] + stats["searched"]
    overall_tp = total_ops / stats["time"] if stats["time"] else float("inf")
    print(
        f"[summary] runs={int(stats['runs'])}, inserted={int(stats['inserted'])}, "
        f"search_queries={int(stats['searched'])}, time={stats['time']:.2f}s, "
        f"throughput={overall_tp:.1f} ops/s"
    )
    log_line = (
        f"mode={args.mode}\t"
        f"dataset={args.dataset or 'synthetic'}\t"
        f"backend={args.backend}\t"
        f"throughput_ops_per_s={overall_tp:.3f}"
    )
    print("[log]", log_line)
    if args.log_file:
        log_path = Path(args.log_file).expanduser()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        needs_header = not log_path.exists() or log_path.stat().st_size == 0
        with log_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            if needs_header:
                writer.writerow(["mode", "dataset", "backend", "throughput_ops_per_s"])
            writer.writerow([args.mode, args.dataset or "synthetic", args.backend, f"{overall_tp:.3f}"])


if __name__ == "__main__":
    main()
