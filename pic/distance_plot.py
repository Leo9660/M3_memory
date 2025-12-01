#!/usr/bin/env python3
"""Plot per-request centroid/neighbor statistics for Agent traces."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from dataset import (
    AgentGymDataset,
    GSM8KReasoningDataset,
    PRMStepwiseDataset,
    UltraChatDataset,
)

try:
    import faiss  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    faiss = None


@dataclass(frozen=True)
class DatasetStats:
    centroid_mean: float
    agent_centroid_mean: float
    agent_similarity_mean: float
    faiss_centroid_mean: float
    adjacent_mean: float
    topk_mean: float
    cluster_distribution: Dict[int, int] = field(default_factory=dict)


DATASET_LOADERS = {
    "agentgym": AgentGymDataset,
    "gsm8k": GSM8KReasoningDataset,
    "prm800k": PRMStepwiseDataset,
    "ultrachat": UltraChatDataset,
}


def dataset_stats_to_json(stats: DatasetStats) -> Dict[str, object]:
    return {
        "centroid_mean": stats.centroid_mean,
        "agent_centroid_mean": stats.agent_centroid_mean,
        "agent_similarity_mean": stats.agent_similarity_mean,
        "faiss_centroid_mean": stats.faiss_centroid_mean,
        "adjacent_mean": stats.adjacent_mean,
        "topk_mean": stats.topk_mean,
        "cluster_distribution": {
            str(cluster_id): count for cluster_id, count in stats.cluster_distribution.items()
        },
    }


def dataset_stats_from_json(data: Mapping[str, object]) -> DatasetStats:
    raw_distribution = data.get("cluster_distribution")
    cluster_distribution: Dict[int, int] = {}
    if isinstance(raw_distribution, Mapping):
        for key, value in raw_distribution.items():
            try:
                cluster_id = int(key)
                cluster_distribution[cluster_id] = int(value)
            except (TypeError, ValueError):
                continue
    return DatasetStats(
        centroid_mean=float(data.get("centroid_mean", float("nan"))),
        agent_centroid_mean=float(data.get("agent_centroid_mean", float("nan"))),
        agent_similarity_mean=float(data.get("agent_similarity_mean", float("nan"))),
        faiss_centroid_mean=float(data.get("faiss_centroid_mean", float("nan"))),
        adjacent_mean=float(data.get("adjacent_mean", float("nan"))),
        topk_mean=float(data.get("topk_mean", float("nan"))),
        cluster_distribution=cluster_distribution,
    )


def build_cache_key(args: argparse.Namespace) -> str:
    payload = {
        "datasets": list(args.datasets),
        "split": args.split,
        "limit": args.limit,
        "max_length": args.max_length,
        "faiss_index": str(Path(args.faiss_index).expanduser().resolve()) if args.faiss_index else None,
        "faiss_topk": args.faiss_topk,
    }
    return json.dumps(payload, sort_keys=True)


def load_cached_stats(cache_path: Path, key: str) -> Optional[Tuple[List[str], Dict[str, DatasetStats]]]:
    try:
        with cache_path.open("r", encoding="utf-8") as fp:
            cache_data = json.load(fp)
    except (OSError, json.JSONDecodeError):
        return None
    entry = cache_data.get(key)
    if not isinstance(entry, Mapping):
        return None
    order = entry.get("order")
    stats_blob = entry.get("stats")
    if not isinstance(order, list) or not isinstance(stats_blob, Mapping):
        return None
    stats_by_dataset: Dict[str, DatasetStats] = {}
    for name in order:
        if not isinstance(name, str) or name not in stats_blob:
            return None
        raw_stats = stats_blob[name]
        if not isinstance(raw_stats, Mapping):
            return None
        stats_by_dataset[name] = dataset_stats_from_json(raw_stats)
    return order, stats_by_dataset


def save_cached_stats(
    cache_path: Path,
    key: str,
    dataset_order: Sequence[str],
    stats_by_dataset: Mapping[str, DatasetStats],
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    data: Dict[str, object] = {}
    if cache_path.is_file():
        try:
            with cache_path.open("r", encoding="utf-8") as fp:
                existing = json.load(fp)
            if isinstance(existing, dict):
                data = existing
        except (OSError, json.JSONDecodeError):
            data = {}
    data[key] = {
        "order": list(dataset_order),
        "stats": {name: dataset_stats_to_json(stats_by_dataset[name]) for name in dataset_order},
    }
    with cache_path.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2)


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    sum_embeddings = (last_hidden_state * mask_expanded).sum(dim=1)
    sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
    return sum_embeddings / sum_mask


def encode_texts(
    texts: Sequence[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    batch_size: int,
    max_length: int | None,
) -> torch.Tensor:
    encoded_batches: List[torch.Tensor] = []
    model.eval()

    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            chunk = texts[start : start + batch_size]
            prefixed = [f"passage: {text}" for text in chunk]
            tokenizer_kwargs: Dict[str, object] = {
                "padding": True,
                "return_tensors": "pt",
                "truncation": True,
            }
            max_len = max_length
            if max_len is None:
                max_len = getattr(tokenizer, "model_max_length", None)
                if not isinstance(max_len, int) or max_len <= 0 or max_len >= 1_000_000:
                    max_len = getattr(model.config, "max_position_embeddings", None)
            if not isinstance(max_len, int) or max_len <= 0:
                max_len = 512
            tokenizer_kwargs["max_length"] = max_len
            inputs = tokenizer(prefixed, **tokenizer_kwargs)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            pooled = mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
            pooled = F.normalize(pooled, p=2, dim=1)
            encoded_batches.append(pooled.cpu())

    return torch.cat(encoded_batches, dim=0)


def extract_faiss_centroids(index) -> np.ndarray | None:
    if faiss is None or index is None:
        return None
    try:
        ivf = faiss.extract_index_ivf(index)
    except AttributeError:
        return None
    if ivf is None:
        return None
    ivf = faiss.downcast_index(ivf)
    quantizer = getattr(ivf, "quantizer", None)
    if quantizer is None:
        return None
    quantizer = faiss.downcast_index(quantizer)
    nlist = int(getattr(ivf, "nlist", 0))
    dim = int(getattr(ivf, "d", 0))
    if quantizer is None or nlist <= 0 or dim <= 0:
        return None
    if hasattr(quantizer, "xb") and getattr(quantizer, "ntotal", 0) == nlist:
        buf = faiss.vector_to_array(quantizer.xb).astype(np.float32, copy=False)
        return buf.reshape(nlist, dim)
    return None


def summarize_faiss_clusters(index) -> tuple[int, List[int]] | None:
    if faiss is None or index is None:
        return None
    try:
        ivf = faiss.extract_index_ivf(index)
    except AttributeError:
        return None
    if ivf is None:
        return None
    ivf = faiss.downcast_index(ivf)
    invlists = getattr(ivf, "invlists", None)
    nlist = int(getattr(ivf, "nlist", 0))
    if invlists is None or nlist <= 0:
        return None

    sizes = [int(invlists.list_size(i)) for i in range(nlist)]
    return nlist, sizes


def trace_to_embeddings(
    trace: Sequence[Mapping[str, object]],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    batch_size: int,
    max_length: int | None,
) -> torch.Tensor:
    texts: List[str] = []
    for item in trace:
        text = str(
            item.get("text")
            or item.get("human")
            or item.get("user")
            or ""
        ).strip()
        if text:
            texts.append(text)
    if not texts:
        return torch.empty((0, model.config.hidden_size))
    return encode_texts(texts, tokenizer, model, device, batch_size, max_length)


def query_faiss_topk(
    embeddings: torch.Tensor,
    index,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    if index is None or embeddings.size(0) == 0:
        return None, None
    requested_k = min(top_k, index.ntotal)
    if requested_k <= 0:
        return None, None

    vectors = embeddings.detach().cpu().numpy().astype(np.float32, copy=False)
    if vectors.shape[1] != index.d:
        raise ValueError(
            f"Embedding dim {vectors.shape[1]} does not match FAISS index dim {index.d}"
        )

    distances, ids = index.search(vectors, requested_k)
    if distances.shape[1] == 0 or ids.shape[1] == 0:
        return None, None
    return distances, ids


def collect_request_statistics(
    data: Sequence[Mapping[str, object]],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    batch_size: int,
    max_length: int | None,
    faiss_centroid_tensor: Optional[torch.Tensor],
    faiss_quantizer,
    faiss_metric_type: Optional[int],
) -> tuple[List[float], List[float], List[int], List[float], float]:
    centroid_records: List[float] = []
    faiss_centroid_records: List[float] = []
    cluster_assignments: List[int] = []
    agent_centroid_records: List[float] = []
    agent_centroids: List[torch.Tensor] = []

    for agent in data:
        requests = agent.get("requests", [])
        agent_chunks: List[torch.Tensor] = []
        for request in requests:
            trace = request.get("trace", [])
            if not trace:
                continue
            embeddings = trace_to_embeddings(trace, tokenizer, model, device, batch_size, max_length)
            if embeddings.size(0) == 0:
                continue

            centroid = embeddings.mean(dim=0, keepdim=True)
            diffs = embeddings - centroid
            centroid_dist = torch.linalg.norm(diffs, dim=1)
            centroid_records.append(float(centroid_dist.mean().item()))
            agent_chunks.append(embeddings.detach().cpu())

            if faiss_centroid_tensor is not None and faiss_centroid_tensor.size(0) > 0:
                item_dists = torch.cdist(embeddings, faiss_centroid_tensor, p=2)
                if item_dists.numel() > 0:
                    min_dists, min_indices = torch.min(item_dists, dim=1)
                    faiss_centroid_records.append(float(min_dists.mean().item()))
                    cluster_assignments.extend(int(idx.item()) for idx in min_indices)
            elif faiss_quantizer is not None:
                vectors = embeddings.detach().cpu().numpy().astype(np.float32, copy=False)
                try:
                    distances, ids = faiss_quantizer.search(vectors, 1)
                except Exception:
                    distances, ids = None, None
                if distances is not None and distances.size > 0:
                    per_item = distances[:, 0]
                    if faiss is not None:
                        metric = faiss_metric_type if faiss_metric_type is not None else getattr(
                            faiss_quantizer, "metric_type", faiss.METRIC_L2
                        )
                        if metric == faiss.METRIC_L2:
                            per_item = np.sqrt(np.maximum(per_item, 0.0))
                    faiss_centroid_records.append(float(per_item.mean()))
                if ids is not None and ids.size > 0:
                    cluster_assignments.extend(
                        int(idx)
                        for idx in ids[:, 0].tolist()
                        if isinstance(idx, (int, np.integer)) and idx >= 0
                    )

        if agent_chunks:
            agent_embeddings = torch.cat(agent_chunks, dim=0)
            agent_centroid = agent_embeddings.mean(dim=0, keepdim=True)
            agent_diffs = agent_embeddings - agent_centroid
            agent_dist = torch.linalg.norm(agent_diffs, dim=1)
            agent_centroid_records.append(float(agent_dist.mean().item()))
            agent_centroids.append(agent_centroid.detach().cpu())

    agent_similarity_mean = float("nan")
    if len(agent_centroids) > 1:
        agent_tensor = torch.cat(agent_centroids, dim=0)
        dist_matrix = torch.cdist(agent_tensor, agent_tensor, p=2)
        mask = torch.ones_like(dist_matrix, dtype=torch.bool)
        mask.fill_diagonal_(False)
        valid = mask.sum()
        if valid > 0:
            agent_similarity_mean = float(dist_matrix[mask].mean().item())

    return (
        centroid_records,
        faiss_centroid_records,
        cluster_assignments,
        agent_centroid_records,
        agent_similarity_mean,
    )


def plot_distance_summary(
    dataset_order: Sequence[str],
    stats_by_dataset: Mapping[str, DatasetStats],
    output: Optional[Path],
) -> None:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(6, 2.3),
        gridspec_kw={"wspace": 0.3},
    )
    ax = axes[0]

    base_x = np.arange(len(dataset_order), dtype=float)
    bar_width = 0.18
    offsets = [-bar_width, 0.0, bar_width]
    centroid_values = [
        stats_by_dataset[name].centroid_mean for name in dataset_order
    ]
    agent_centroid_values = [
        stats_by_dataset[name].agent_centroid_mean for name in dataset_order
    ]
    faiss_centroid_values = [
        stats_by_dataset[name].faiss_centroid_mean for name in dataset_order
    ]

    centroid_bars = ax.bar(
        base_x + offsets[0],
        centroid_values,
        width=bar_width,
        color="#5b8cc5",
        edgecolor="#264b73",
        linewidth=0.9,
        label="Intra-request centroid",
    )
    agent_centroid_bars = ax.bar(
        base_x + offsets[1],
        agent_centroid_values,
        width=bar_width,
        color="#7dc8b1",
        edgecolor="#2f6b5b",
        linewidth=0.9,
        label="Inter-request centroid",
    )
    faiss_bars = ax.bar(
        base_x + offsets[2],
        faiss_centroid_values,
        width=bar_width,
        color="#d46a6a",
        edgecolor="#7a2f2f",
        linewidth=0.9,
        label="Old centroid",
    )

    ax.set_xticks(base_x)
    ax.set_xticklabels(dataset_order, fontsize=9)
    ax.set_ylabel("L2 distance", fontsize=9)
    ax.set_title("(a) Similarity with different centroids", fontsize=9)
    ax.set_ylim(0, 0.85)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.tick_params(axis="y", labelsize=9)
    ax.tick_params(axis="x", labelsize=9)
    ax.margins(x=0.12)

    cluster_ax = axes[1]
    has_cluster_data = any(
        stats_by_dataset[name].cluster_distribution for name in dataset_order
    )
    if not has_cluster_data:
        cluster_ax.axis("off")
        cluster_ax.text(
            0.5,
            0.5,
            "No FAISS centroid clusters available",
            ha="center",
            va="center",
            fontsize=9,
            color="#5f6368",
        )
    else:
        bar_color = "#c7edc7"
        max_segments = 50
        x_positions: List[float] = []
        heights: List[float] = []
        widths: List[float] = []
        dataset_low_labels: List[Tuple[float, float]] = []
        group_labels: List[str] = []
        group_centers: List[float] = []
        current_pos = 0.0
        max_percentage = 0.0
        low_threshold = 5.0
        low_count_total = 0
        red_boxes: List[Rectangle] = []

        for dataset in dataset_order:
            counts = stats_by_dataset[dataset].cluster_distribution
            total_assignments = sum(counts.values())
            if total_assignments <= 0 or not counts:
                group_labels.append(f"{dataset}\nN/A")
                group_centers.append(current_pos)
                current_pos += 1.0
                continue

            low_mass_total = sum(
                count for count in counts.values()
                if (count / total_assignments) * 100.0 <= low_threshold
            )
            low_fraction = (low_mass_total / total_assignments) * 100.0 if total_assignments > 0 else 0.0

            sorted_clusters = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:max_segments]
            num_clusters = len(sorted_clusters)

            if num_clusters <= 0:
                continue

            local_x = np.linspace(current_pos, current_pos + 1.0, num_clusters, endpoint=False)
            bar_w = 1.0 / num_clusters
            dataset_positions: List[float] = []
            dataset_lows: List[int] = []
            for idx, (cluster_id, count) in enumerate(sorted_clusters):
                percentage = (count / total_assignments) * 100.0
                center_x = local_x[idx] + bar_w / 2
                x_positions.append(center_x)
                widths.append(bar_w)
                heights.append(percentage)
                dataset_positions.append(center_x)
                max_percentage = max(max_percentage, percentage)
                if percentage <= low_threshold:
                    low_count_total += 1
                    dataset_lows.append(idx)

            if dataset_lows:
                first_idx = min(dataset_lows)
                last_idx = max(dataset_lows)
                left = dataset_positions[first_idx] - bar_w / 2
                right = dataset_positions[last_idx] + bar_w / 2
                red_boxes.append(
                    Rectangle((left, 0), right - left, low_threshold, fill=False, edgecolor="red", linewidth=1.0)
                )
                dataset_low_labels.append((current_pos + 0.5, low_fraction))
            else:
                dataset_low_labels.append((current_pos + 0.5, low_fraction))

            group_centers.append(current_pos + 0.5)
            group_labels.append(dataset)
            current_pos = current_pos + 1.0 + 0.4

        if not x_positions:
            cluster_ax.axis("off")
        else:
            cluster_ax.bar(
                x_positions,
                heights,
                width=widths if widths else 1.0,
                color=bar_color,
                edgecolor="#5f8f5f",
                linewidth=0.4,
            )
            for box in red_boxes:
                cluster_ax.add_patch(box)
            for x_pos, pct_sum in dataset_low_labels:
                cluster_ax.annotate(
                    f"{pct_sum:.1f}%",
                    xy=(x_pos, 5),
                    xycoords="data",
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="red",
                    arrowprops=dict(arrowstyle="->", color="red", lw=0.6),
                )
            cluster_ax.set_xticks(group_centers)
            cluster_ax.set_xticklabels(group_labels, rotation=0, fontsize=9)
            cluster_ax.set_xlim(-0.2, current_pos)
            ylim = max_percentage * 1.1 if max_percentage > 0 else 1.0
            cluster_ax.set_ylim(0, ylim)
            cluster_ax.set_ylabel("Frequency (%)", fontsize=9)
            cluster_ax.set_title("(b) Cluster Assignment (sorted)", fontsize=9)
            cluster_ax.grid(axis="y", linestyle="--", alpha=0.3)
            cluster_ax.tick_params(axis="y", labelsize=9)
            cluster_ax.tick_params(axis="x", labelsize=9)

            # legend_handles: List[Patch] = []
            # legend_handles.append(Patch(color=bar_color, label="Cluster frequency"))
            # cluster_ax.legend(
            #     handles=legend_handles,
            #     loc="upper right",
            #     fontsize=7,
            #     frameon=False,
            # )
            # if red_boxes:
            #     first_box = red_boxes[0]
            #     box_x, box_y = first_box.get_xy()
            #     cluster_ax.annotate(
            #         f"≤{low_threshold:.0f}% segments: {low_count_total}",
            #         xy=(box_x + first_box.get_width() / 2, box_y + first_box.get_height() * 0.8),
            #         xycoords="data",
            #         xytext=(0, -22),
            #         textcoords="offset points",
            #         ha="center",
            #         va="top",
            #         fontsize=7,
            #         color="red",
            #         arrowprops=dict(arrowstyle="->", color="red", lw=0.7),
            #     )

    ax.legend(
        handles=[centroid_bars, agent_centroid_bars, faiss_bars],
        loc="upper left",
        fontsize=8,
        frameon=False,
    )

    # fig.suptitle("Dataset-level Euclidean distance overview", fontsize=16)
    # fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.tight_layout(pad=0.45, w_pad=0.4, h_pad=0.4)

    if output:
        fig.savefig(output, dpi=300)
    else:
        plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot agent request centroid and neighbor distances.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASET_LOADERS.keys()),
        default=["prm800k", "ultrachat"],
        help="Datasets to load (defaults to PRM800k + UltraChat).",
    )
    parser.add_argument("--split", default=None, help="Dataset split to load.")
    parser.add_argument("--limit", type=int, default=None, help="Limit requests per agent.")
    parser.add_argument("--cache-dir", default=None, help="Optional Hugging Face cache directory.")
    parser.add_argument("--batch-size", type=int, default=8, help="Embedding batch size.")
    parser.add_argument("--max-length", type=int, default=None, help="Tokenizer truncation length.")
    parser.add_argument(
        "--device",
        default=None,
        help="torch device to run on (defaults to cuda if available).",
    )
    parser.add_argument(
        "--faiss-index",
        default=None,
        help="Path to a FAISS index for nearest-neighbor lookups (required for FAISS bars).",
    )
    parser.add_argument(
        "--faiss-topk",
        type=int,
        default=1,
        help="Number of closest FAISS centroids to average per request (>=1).",
    )
    parser.add_argument(
        "--faiss-nprobe",
        type=int,
        default=32,
        help="nprobe value to configure on the FAISS IVF index.",
    )
    parser.add_argument(
        "--cache-file",
        type=Path,
        default=Path("pic/distance_stats_cache.json"),
        help="JSON file for cached statistics (set to empty string to disable).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable reading/writing cached statistics.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Path to save the figure.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_path: Optional[Path] = None
    if args.cache_file:
        cache_path = Path(args.cache_file).expanduser()
    cache_key = build_cache_key(args)

    stats_by_dataset: Dict[str, DatasetStats] = {}
    dataset_order: List[str] = []
    used_cache = False
    if (
        not args.no_cache
        and cache_path is not None
        and cache_path.is_file()
    ):
        cached = load_cached_stats(cache_path, cache_key)
        if cached is not None:
            dataset_order, stats_by_dataset = cached
            used_cache = True

    if not used_cache:
        device = torch.device(
            args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        tokenizer = AutoTokenizer.from_pretrained("intfloat/e5-large-v2")
        model = AutoModel.from_pretrained("intfloat/e5-large-v2").to(device)

        faiss_index = None
        faiss_centroid_tensor: Optional[torch.Tensor] = None
        faiss_quantizer = None
        faiss_metric_type: Optional[int] = None
        if args.faiss_index:
            if faiss is None:
                raise SystemExit("faiss library is not installed; cannot use --faiss-index.")
            if args.faiss_topk <= 0:
                raise SystemExit("--faiss-topk must be > 0 when using --faiss-index.")
            if args.faiss_nprobe <= 0:
                raise SystemExit("--faiss-nprobe must be > 0 when using --faiss-index.")
            faiss_index = faiss.read_index(args.faiss_index)
            if hasattr(faiss_index, "nprobe"):
                faiss_index.nprobe = args.faiss_nprobe
            faiss_metric_type = getattr(faiss_index, "metric_type", None)
            ivf_index = None
            try:
                ivf_index = faiss.extract_index_ivf(faiss_index)
            except Exception:
                ivf_index = None
            if ivf_index is not None:
                quantizer = getattr(ivf_index, "quantizer", None)
                if quantizer is not None:
                    faiss_quantizer = faiss.downcast_index(quantizer)
            centroids = extract_faiss_centroids(faiss_index)
            if centroids is not None and centroids.size > 0:
                faiss_centroid_tensor = torch.from_numpy(centroids.astype(np.float32, copy=False))
            cluster_summary = summarize_faiss_clusters(faiss_index)
            if cluster_summary is not None:
                nlist, sizes = cluster_summary
                top_sizes = sorted(sizes, reverse=True)[:10]
                print(
                    f"FAISS index clusters: {nlist}; top-10 cluster sizes: {top_sizes}"
                )
        else:
            print("Warning: --faiss-index not provided; FAISS centroid bars will be NaN.")

        for dataset_name in args.datasets:
            dataset_cls = DATASET_LOADERS[dataset_name]
            dataset_kwargs = {"cache_dir": args.cache_dir, "limit": args.limit}
            if args.split:
                dataset_kwargs["split"] = args.split
            dataset = dataset_cls(**dataset_kwargs)
            data = dataset.get_data()
            if not data:
                print(f"Dataset {dataset_name} produced no data; skipping.")
                continue

            (
                centroid_records,
                faiss_centroid_records,
                cluster_assignments,
                agent_centroid_records,
                agent_similarity_mean,
            ) = collect_request_statistics(
                data,
                tokenizer,
                model,
                device,
                args.batch_size,
                args.max_length,
                faiss_centroid_tensor,
                faiss_quantizer,
                faiss_metric_type,
            )

            centroid_mean = float(np.mean(centroid_records)) if centroid_records else float("nan")
            agent_centroid_mean = (
                float(np.mean(agent_centroid_records)) if agent_centroid_records else float("nan")
            )
            agent_similarity_mean = float(agent_similarity_mean)
            faiss_centroid_mean = (
                float(np.mean(faiss_centroid_records)) if faiss_centroid_records else float("nan")
            )
            adjacent_mean = float("nan")
            topk_mean = float("nan")
            cluster_distribution: Dict[int, int] = {}
            if cluster_assignments:
                cluster_counts = Counter(cluster_assignments)
                cluster_distribution = dict(cluster_counts)
            stats_by_dataset[dataset_name] = DatasetStats(
                centroid_mean=centroid_mean,
                agent_centroid_mean=agent_centroid_mean,
                agent_similarity_mean=agent_similarity_mean,
                faiss_centroid_mean=faiss_centroid_mean,
                adjacent_mean=adjacent_mean,
                topk_mean=topk_mean,
                cluster_distribution=cluster_distribution,
            )
            dataset_order.append(dataset_name)

        if not args.no_cache and cache_path is not None and stats_by_dataset:
            save_cached_stats(cache_path, cache_key, dataset_order, stats_by_dataset)

    if not stats_by_dataset:
        raise SystemExit("No datasets produced statistics; nothing to plot.")

    cluster_ids = set()
    for name in dataset_order:
        stats = stats_by_dataset[name]
        dataset_cluster_ids = set(stats.cluster_distribution.keys())
        cluster_ids.update(dataset_cluster_ids)
        print(f"{name}: {len(dataset_cluster_ids)} clusters with assignments")
    print(f"Total clusters with assignments (union): {len(cluster_ids)}")

    plot_distance_summary(
        dataset_order,
        stats_by_dataset,
        args.output,
    )


if __name__ == "__main__":
    main()
