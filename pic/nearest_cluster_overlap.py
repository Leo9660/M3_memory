#!/usr/bin/env python3
"""
Given a FAISS IVF index and two datasets from dataset/ loaders, assign each
text span to its nearest cluster and find the overlap sorted by combined count.

The script:
    1) Loads two datasets via the DatasetBase implementations (same choices as
       agent_similarity.py: agentgym/prm800k/ultrachat/gsm8k/ultrafeedback/xlam).
    2) Encodes every text span in request traces with intfloat/e5-large-v2
       (default; configurable). Optionally cap total items across both datasets
       with --total-limit (roughly half from each; leftover flows to B).
    3) Uses the IVF quantizer to find the nearest centroid (top-1) for each
       embedding and counts occurrences per cluster.
    4) Intersects the two cluster sets and reports clusters ranked by
       min(count_a, count_b) desc, printing counts per cluster and optional CSV.
    5) Downsamples both datasets to the smaller side within the top-3 overlap
       clusters, runs a 2D PCA, and plots the projected points (optional).

Example:
    python pic/nearest_cluster_overlap.py \\
        --faiss-index /path/to/index.faiss \\
        --dataset-a agentgym --dataset-b ultrachat \\
        --total-limit 2000 \\
        --topk 30 --output pic/cluster_overlap.csv
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Sequence, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from transformers import AutoModel, AutoTokenizer

from dataset import (
    AgentGymDataset,
    GSM8KReasoningDataset,
    PRMStepwiseDataset,
    UltraChatDataset,
    UltraFeedbackDataset,
    XLAMFunctionCallingDataset,
)

try:
    import faiss  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise SystemExit("faiss is required to run this script.") from exc


DATASET_CHOICES = {
    "agentgym": AgentGymDataset,
    "prm800k": PRMStepwiseDataset,
    "ultrachat": UltraChatDataset,
    "gsm8k": GSM8KReasoningDataset,
    "ultrafeedback": UltraFeedbackDataset,
    "xlam": XLAMFunctionCallingDataset,
}


def encode_texts(
    texts: Sequence[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    batch_size: int,
    max_length: int | None,
) -> np.ndarray:
    outputs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = tokenizer(
                [f"query: {t}" for t in batch],
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=max_length,
            ).to(device)
            last_hidden = model(**inputs).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1)
            pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            outputs.append(pooled.cpu().numpy())
    return np.concatenate(outputs, axis=0) if outputs else np.empty((0, model.config.hidden_size))


def _iter_texts(data: Sequence[Mapping[str, object]], limit: int | None = None) -> Iterable[str]:
    count = 0
    for agent in data:
        for req in agent.get("requests", []) or []:
            for item in req.get("trace", []) or []:
                text = (item.get("text") or "").strip()
                if not text:
                    continue
                yield text
                count += 1
                if limit is not None and count >= limit:
                    return


def _assign_clusters(
    quantizer,
    texts: Iterable[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    batch_size: int,
    max_length: int | None,
) -> tuple[Counter, int]:
    counter: Counter = Counter()
    buffer: list[str] = []
    total = 0
    for text in texts:
        buffer.append(text)
        if len(buffer) >= batch_size * 4:  # slightly larger chunk to amortize search
            embeddings = encode_texts(buffer, tokenizer, model, device, batch_size, max_length)
            _, ids = quantizer.search(embeddings, 1)
            counter.update(int(cid) for cid in ids[:, 0] if cid >= 0)
            total += len(buffer)
            buffer.clear()
    if buffer:
        embeddings = encode_texts(buffer, tokenizer, model, device, batch_size, max_length)
        _, ids = quantizer.search(embeddings, 1)
        counter.update(int(cid) for cid in ids[:, 0] if cid >= 0)
        total += len(buffer)
    return counter, total


def _assign_and_collect(
    quantizer,
    texts: Iterable[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    batch_size: int,
    max_length: int | None,
    keep_clusters: set[int],
    limit: int | None,
) -> tuple[Counter, int, list[tuple[np.ndarray, int]]]:
    counter: Counter = Counter()
    buffer: list[str] = []
    collected: list[tuple[np.ndarray, int]] = []
    total = 0
    for text in texts:
        buffer.append(text)
        if len(buffer) >= batch_size * 4:
            embeddings = encode_texts(buffer, tokenizer, model, device, batch_size, max_length)
            _, ids = quantizer.search(embeddings, 1)
            counter.update(int(cid) for cid in ids[:, 0] if cid >= 0)
            for emb, cid in zip(embeddings, ids[:, 0]):
                cid_int = int(cid)
                if cid_int in keep_clusters:
                    collected.append((emb, cid_int))
            total += len(buffer)
            buffer.clear()
            if limit is not None and total >= limit:
                break
    if buffer and (limit is None or total < limit):
        embeddings = encode_texts(buffer, tokenizer, model, device, batch_size, max_length)
        _, ids = quantizer.search(embeddings, 1)
        counter.update(int(cid) for cid in ids[:, 0] if cid >= 0)
        for emb, cid in zip(embeddings, ids[:, 0]):
            cid_int = int(cid)
            if cid_int in keep_clusters:
                collected.append((emb, cid_int))
        total += len(buffer)
    return counter, total, collected


def _load_dataset(name: str, split: str | None, cache_dir: str | None, limit: int | None):
    cls = DATASET_CHOICES.get(name)
    if cls is None:
        raise ValueError(f"Unsupported dataset {name}")
    kwargs = {}
    if split:
        kwargs["split"] = split
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    # Datasets that support limit use "limit" kwarg; others will ignore via **kwargs
    if limit is not None:
        kwargs["limit"] = limit
    try:
        return cls(**kwargs)  # type: ignore[arg-type]
    except TypeError:
        kwargs.pop("limit", None)
        return cls(**kwargs)  # type: ignore[arg-type]


def pca_fit_transform(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if matrix.shape[0] == 0:
        return np.empty((0, 2)), np.zeros((1, matrix.shape[1] if matrix.ndim == 2 else 0)), np.zeros((matrix.shape[1], 2) if matrix.ndim == 2 else (0, 2))
    mean = matrix.mean(axis=0, keepdims=True)
    centered = matrix - mean
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    components = vh[:2].T
    coords = centered @ components
    return coords, mean, components


def _extract_ivf_vectors(ivf, list_id: int, max_items: int = 2000) -> np.ndarray:
    invlists = getattr(ivf, "invlists", None)
    if invlists is None:
        return np.empty((0, int(ivf.d)), dtype=np.float32)
    size = int(invlists.list_size(list_id))
    if size <= 0:
        return np.empty((0, int(ivf.d)), dtype=np.float32)
    code_size = int(getattr(ivf, "code_size", 0))
    dim = int(ivf.d)
    if code_size != dim * 4:
        # Only support IVFFlat-style raw float codes
        return np.empty((0, dim), dtype=np.float32)
    codes_ptr = invlists.get_codes(list_id)
    if codes_ptr is None:
        return np.empty((0, dim), dtype=np.float32)
    codes = faiss.rev_swig_ptr(codes_ptr, size * code_size)
    codes = np.asarray(codes, dtype=np.uint8)
    float_view = codes.view(np.float32).reshape(size, dim)
    if size > max_items:
        rng = np.random.default_rng(123)
        idx = rng.choice(size, size=max_items, replace=False)
        float_view = float_view[idx]
    return float_view.astype(np.float32, copy=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find overlapping nearest clusters between two datasets for an IVF FAISS index (using dataset loaders + encoder)."
    )
    parser.add_argument("--faiss-index", required=True, help="Path to FAISS IVF index.")
    parser.add_argument("--dataset-a", choices=DATASET_CHOICES.keys(), required=True, help="Dataset name for side A.")
    parser.add_argument("--dataset-b", choices=DATASET_CHOICES.keys(), required=True, help="Dataset name for side B.")
    parser.add_argument("--split-a", default=None, help="Optional split for dataset A.")
    parser.add_argument("--split-b", default=None, help="Optional split for dataset B.")
    parser.add_argument("--total-limit", type=int, default=None, help="Total text items to sample (roughly half from each dataset; leftover flows to B if A is short).")
    parser.add_argument("--cache-dir", default=None, help="Hugging Face cache dir for datasets/tokenizer/model.")
    parser.add_argument("--model-name", default="intfloat/e5-large-v2", help="Encoder model name.")
    parser.add_argument("--batch-size", type=int, default=8, help="Encoding batch size.")
    parser.add_argument("--max-length", type=int, default=512, help="Tokenizer max_length.")
    parser.add_argument("--device", default=None, help="torch device (default: cuda if available else cpu).")
    parser.add_argument("--nprobe", type=int, default=32, help="nprobe to set on the IVF index.")
    parser.add_argument("--topk", type=int, default=20, help="How many clusters to display/save (sorted by combined count).")
    parser.add_argument("--output", default=None, help="Optional CSV output path.")
    parser.add_argument("--plot", default=None, help="Optional PNG/PDF path for 2D PCA scatter of top-3 overlap clusters.")
    parser.add_argument("--pca-seed", type=int, default=42, help="RNG seed for PCA downsampling.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    index_path = Path(args.faiss_index).expanduser().resolve()
    if not index_path.is_file():
        raise FileNotFoundError(f"Index not found: {index_path}")

    index = faiss.read_index(str(index_path))
    ivf = faiss.extract_index_ivf(index)
    if ivf is None:
        raise ValueError("Index does not contain an IVF component.")
    ivf = faiss.downcast_index(ivf)
    quantizer = faiss.downcast_index(ivf.quantizer)
    ivf.nprobe = int(args.nprobe)
    dim = int(ivf.d)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    model = AutoModel.from_pretrained(args.model_name, cache_dir=args.cache_dir).to(device)
    if model.config.hidden_size != dim:
        raise ValueError(f"Model hidden size {model.config.hidden_size} != index dim {dim}")

    ds_a = _load_dataset(args.dataset_a, args.split_a, args.cache_dir, None)
    ds_b = _load_dataset(args.dataset_b, args.split_b, args.cache_dir, None)
    data_a = ds_a.get_data()
    data_b = ds_b.get_data()

    target_a = target_b = None
    if args.total_limit is not None and args.total_limit > 0:
        target_a = args.total_limit // 2
        target_b = args.total_limit - target_a

    counts_a = _assign_clusters(
        quantizer,
        _iter_texts(data_a, target_a),
        tokenizer,
        model,
        device,
        args.batch_size,
        args.max_length,
    )
    counts_a, processed_a = counts_a
    remaining_for_b = None
    if args.total_limit is not None and args.total_limit > 0:
        remaining_for_b = max(args.total_limit - processed_a, 0)

    counts_b = _assign_clusters(
        quantizer,
        _iter_texts(data_b, remaining_for_b if remaining_for_b is not None else target_b),
        tokenizer,
        model,
        device,
        args.batch_size,
        args.max_length,
    )
    counts_b, processed_b = counts_b

    overlap: dict[int, Tuple[int, int, int]] = {}
    for cid in counts_a.keys() & counts_b.keys():
        c1 = counts_a[cid]
        c2 = counts_b[cid]
        overlap[cid] = (c1, c2, c1 + c2)

    ranked = sorted(overlap.items(), key=lambda kv: min(kv[1][0], kv[1][1]), reverse=True)
    if args.topk is not None and args.topk > 0:
        ranked = ranked[: args.topk]

    print(f"Index: {index_path} (nlist={getattr(ivf, 'nlist', 'N/A')})")
    print(f"Dataset A: {args.dataset_a} (agents={len(data_a)}, texts_used={processed_a})")
    print(f"Dataset B: {args.dataset_b} (agents={len(data_b)}, texts_used={processed_b})")
    if args.total_limit:
        print(f"Total text cap: {args.total_limit} (A target≈{target_a}, B target≈{target_b}, actual={processed_a + processed_b})")
    print(f"Clusters in both: {len(overlap)}")
    for cid, (c1, c2, _) in ranked:
        print(f"cluster={cid}: A={c1}, B={c2}, combined={c1 + c2}, min={min(c1, c2)}")

    if ranked and args.plot:
        top_clusters = {cid for cid, _ in ranked[: min(3, len(ranked))]}  # top-3 by min count
        rng = np.random.default_rng(args.pca_seed)

        _, _, embeds_a = _assign_and_collect(
            quantizer,
            _iter_texts(data_a, target_a),
            tokenizer,
            model,
            device,
            args.batch_size,
            args.max_length,
            top_clusters,
            target_a,
        )
        _, _, embeds_b = _assign_and_collect(
            quantizer,
            _iter_texts(data_b, remaining_for_b if remaining_for_b is not None else target_b),
            tokenizer,
            model,
            device,
            args.batch_size,
            args.max_length,
            top_clusters,
            remaining_for_b if remaining_for_b is not None else target_b,
        )

        # per-cluster downsample to balance A/B; include IVF vectors per cluster
        cluster_list = sorted(top_clusters)
        a_vecs_all: list[np.ndarray] = []
        b_vecs_all: list[np.ndarray] = []
        ivf_vecs_all: list[np.ndarray] = []
        a_clusters: list[int] = []
        b_clusters: list[int] = []
        ivf_clusters: list[int] = []

        for cid in cluster_list:
            a_vecs = np.array([e for e, c in embeds_a if c == cid], dtype=np.float32)
            b_vecs = np.array([e for e, c in embeds_b if c == cid], dtype=np.float32)
            if len(a_vecs) == 0 or len(b_vecs) == 0:
                continue
            min_len = min(len(a_vecs), len(b_vecs))
            a_idx = rng.choice(len(a_vecs), size=min_len, replace=False) if len(a_vecs) > min_len else np.arange(len(a_vecs))
            b_idx = rng.choice(len(b_vecs), size=min_len, replace=False) if len(b_vecs) > min_len else np.arange(len(b_vecs))
            a_vecs = a_vecs[a_idx]
            b_vecs = b_vecs[b_idx]
            a_vecs_all.append(a_vecs)
            b_vecs_all.append(b_vecs)
            a_clusters.extend([cid] * len(a_vecs))
            b_clusters.extend([cid] * len(b_vecs))

            ivf_vecs = _extract_ivf_vectors(ivf, int(cid))
            if ivf_vecs.shape[0] > 0:
                ivf_vecs_all.append(ivf_vecs)
                ivf_clusters.extend([cid] * len(ivf_vecs))

        if a_vecs_all and b_vecs_all:
            a_vecs = np.vstack(a_vecs_all)
            b_vecs = np.vstack(b_vecs_all)
            ivf_vecs = np.vstack(ivf_vecs_all) if ivf_vecs_all else np.empty((0, dim), dtype=np.float32)

            all_vecs = np.vstack([arr for arr in (a_vecs, b_vecs, ivf_vecs) if arr.size > 0])
            coords, mean_vec, components = pca_fit_transform(all_vecs)

            len_a = len(a_vecs)
            len_b = len(b_vecs)
            coords_a = coords[:len_a]
            coords_b = coords[len_a:len_a + len_b]
            coords_ivf = coords[len_a + len_b:] if ivf_vecs.size > 0 else np.empty((0, 2))

            a_clusters_arr = np.array(a_clusters, dtype=int)
            b_clusters_arr = np.array(b_clusters, dtype=int)
            ivf_clusters_arr = np.array(ivf_clusters, dtype=int) if ivf_vecs.size > 0 else np.empty((0,), dtype=int)

            fig, ax = plt.subplots(figsize=(5, 4))
            # marker size smaller when many points
            base_size = max(2.5, 12.0 * (200.0 / max(len_a, len_b, 200)))
            colors = ["#1f77b4", "#2ca02c", "#9467bd"]
            markers = ["o", "s", "D"]
            for idx, cid in enumerate(cluster_list):
                color = colors[idx % len(colors)]
                marker = markers[idx % len(markers)]
                mask_a = a_clusters_arr == cid
                mask_b = b_clusters_arr == cid
                if mask_a.any():
                    ax.scatter(
                        coords_a[mask_a, 0],
                        coords_a[mask_a, 1],
                        marker=marker,
                        s=base_size,
                        c=color,
                        alpha=0.7,
                        label=f"A cluster {cid}",
                    )
                if mask_b.any():
                    ax.scatter(
                        coords_b[mask_b, 0],
                        coords_b[mask_b, 1],
                        marker=marker,
                        s=base_size,
                        facecolors="none",
                        edgecolors=color,
                        alpha=0.8,
                        label=f"B cluster {cid}",
                    )
                if coords_ivf.shape[0] > 0:
                    mask_ivf = ivf_clusters_arr == cid
                    if mask_ivf.any():
                        ax.scatter(
                            coords_ivf[mask_ivf, 0],
                            coords_ivf[mask_ivf, 1],
                            marker=".",
                            s=max(2.0, base_size * 0.4),
                            c=color,
                            alpha=0.3,
                            label=f"IVF vectors {cid}",
                        )
                try:
                    centroid = quantizer.reconstruct(int(cid))
                    centroid = np.asarray(centroid, dtype=np.float32)
                    if centroid.shape[0] == components.shape[0]:
                        projected = (centroid - mean_vec.squeeze(0)) @ components
                        ax.scatter(
                            projected[0],
                            projected[1],
                            marker="*",
                            s=80,
                            c=color,
                            edgecolors="#000000",
                            linewidths=0.6,
                            alpha=0.95,
                            label=f"Centroid {cid}",
                        )
                except Exception:
                    pass

            ax.set_title("Top overlap clusters PCA (downsampled)")
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            handles, labels = ax.get_legend_handles_labels()
            seen = {}
            uniq_handles = []
            uniq_labels = []
            for h, l in zip(handles, labels):
                if l in seen:
                    continue
                seen[l] = True
                uniq_handles.append(h)
                uniq_labels.append(l)
            ax.legend(uniq_handles, uniq_labels, fontsize=8, frameon=False)
            ax.grid(True, linestyle="--", alpha=0.3)

            out_plot = Path(args.plot).expanduser()
            out_plot.parent.mkdir(parents=True, exist_ok=True)
            fig.tight_layout()
            fig.savefig(out_plot, dpi=300)
            plt.close(fig)
            print(f"Saved PCA plot to {out_plot}")

    if args.output:
        out_path = Path(args.output).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.writer(fp)
            writer.writerow(["cluster_id", "count_data_a", "count_data_b", "combined"])
            for cid, (c1, c2, combined) in ranked:
                writer.writerow([cid, c1, c2, combined])
        print(f"Saved CSV to {out_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
