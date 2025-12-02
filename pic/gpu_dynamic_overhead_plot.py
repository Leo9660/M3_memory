#!/usr/bin/env python3
"""Synthetic CPU vs GPU similarity timing and dynamic resize overhead plot."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - optional GPU timing
    torch = None


def power_of_two_sizes(min_power: int, max_power: int) -> List[int]:
    if min_power > max_power:
        raise ValueError("--min-power must be <= --max-power")
    return [2 ** p for p in range(min_power, max_power + 1)]


def benchmark_cpu_similarity(vecs: np.ndarray, query: np.ndarray, repeats: int, warmup: int) -> float:
    durations: List[float] = []
    for i in range(warmup + repeats):
        start = time.perf_counter()
        _ = vecs @ query  # shape: (cluster_size,)
        end = time.perf_counter()
        if i >= warmup:
            durations.append((end - start) * 1000.0)
    return float(np.median(durations)) if durations else float("nan")


def benchmark_gpu_similarity(
    vecs: np.ndarray,
    query: np.ndarray,
    repeats: int,
    warmup: int,
) -> float | None:
    if torch is None or not torch.cuda.is_available():
        return None
    device = torch.device("cuda")
    vecs_t = torch.from_numpy(vecs).to(device)
    query_t = torch.from_numpy(query).to(device)
    torch.cuda.synchronize()
    for _ in range(warmup):
        _ = torch.matmul(vecs_t, query_t)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        _ = torch.matmul(vecs_t, query_t)
    torch.cuda.synchronize()
    end = time.perf_counter()
    return (end - start) / max(repeats, 1) * 1000.0


def benchmark_gpu_transfer(vecs: np.ndarray, repeats: int, warmup: int) -> float | None:
    if torch is None or not torch.cuda.is_available():
        return None
    device = torch.device("cuda")
    durations: List[float] = []
    for i in range(warmup + repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        _ = torch.from_numpy(vecs).to(device, non_blocking=True)
        torch.cuda.synchronize()
        end = time.perf_counter()
        if i >= warmup:
            durations.append((end - start) * 1000.0)
    return float(np.median(durations)) if durations else None


def benchmark_gpu_to_gpu(vecs: np.ndarray, repeats: int, warmup: int) -> float | None:
    if torch is None or not torch.cuda.is_available():
        return None
    device = torch.device("cuda")
    base = torch.from_numpy(vecs).to(device)
    durations: List[float] = []
    for i in range(warmup + repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        clone = torch.empty_like(base)
        clone.copy_(base)
        torch.cuda.synchronize()
        end = time.perf_counter()
        if i >= warmup:
            durations.append((end - start) * 1000.0)
        del clone
    del base
    return float(np.median(durations)) if durations else None


def benchmark_gpu_full_init(vecs: np.ndarray, repeats: int, warmup: int) -> float | None:
    if torch is None or not torch.cuda.is_available():
        return None
    device = torch.device("cuda")
    durations: List[float] = []
    for i in range(warmup + repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        _ = torch.empty((vecs.shape[0], vecs.shape[1]), device=device)
        torch.cuda.synchronize()
        end = time.perf_counter()
        if i >= warmup:
            durations.append((end - start) * 1000.0)
    return float(np.median(durations)) if durations else None


def simulate_gpu_time(cpu_ms: float, speedup: float) -> float:
    if speedup <= 0:
        return cpu_ms
    return cpu_ms / speedup


def simulate_transfer_costs(
    bytes_total: int,
    *,
    pcie_gbps: float,
    gpu_copy_gbps: float,
    full_init_overhead_ms: float,
    gpu_copy_overhead_ms: float,
) -> Tuple[float, float, float]:
    transfer_ms = (bytes_total / (pcie_gbps * 1e9)) * 1000.0
    full_init_ms = full_init_overhead_ms  # allocation only (no copy)
    gpu_gpu_ms = (bytes_total / (gpu_copy_gbps * 1e9)) * 1000.0 + gpu_copy_overhead_ms
    return full_init_ms, transfer_ms, gpu_gpu_ms


def collect_timings(
    cluster_sizes: Sequence[int],
    dim: int,
    repeats: int,
    warmup: int,
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> Tuple[List[float], List[float], List[float], List[float], List[float]]:
    cpu_times: List[float] = []
    gpu_times: List[float] = []
    full_init_times: List[float] = []
    cpu_gpu_transfer_times: List[float] = []
    gpu_gpu_transfer_times: List[float] = []

    for size in cluster_sizes:
        vecs = rng.standard_normal((size, dim), dtype=np.float32)
        query = rng.standard_normal((dim,), dtype=np.float32)
        cpu_ms = benchmark_cpu_similarity(vecs, query, repeats, warmup)
        gpu_ms = benchmark_gpu_similarity(vecs, query, repeats, warmup)

        if gpu_ms is None:
            gpu_ms = simulate_gpu_time(cpu_ms, args.simulated_gpu_speedup)

        full_init_ms = benchmark_gpu_full_init(vecs, repeats, warmup)
        transfer_ms = benchmark_gpu_transfer(vecs, repeats, warmup)
        gpu_gpu_ms = benchmark_gpu_to_gpu(vecs, repeats, warmup)

        if full_init_ms is None or transfer_ms is None or gpu_gpu_ms is None:
            full_init_ms, transfer_ms, gpu_gpu_ms = simulate_transfer_costs(
                vecs.nbytes,
                pcie_gbps=args.pcie_gbps,
                gpu_copy_gbps=args.gpu_copy_gbps,
                full_init_overhead_ms=args.full_init_overhead_ms,
                gpu_copy_overhead_ms=args.resize_overhead_ms,
            )
            gpu_gpu_ms = (vecs.nbytes / (args.gpu_copy_gbps * 1e9)) * 1000.0 + args.resize_overhead_ms

        cpu_times.append(cpu_ms)
        gpu_times.append(gpu_ms)
        full_init_times.append(full_init_ms)
        cpu_gpu_transfer_times.append(transfer_ms)
        gpu_gpu_transfer_times.append(gpu_gpu_ms)

    return cpu_times, gpu_times, full_init_times, cpu_gpu_transfer_times, gpu_gpu_transfer_times


def plot_results(
    cluster_sizes: Sequence[int],
    cpu_times: Sequence[float],
    gpu_times: Sequence[float],
    full_init_times: Sequence[float],
    cpu_gpu_transfer_times: Sequence[float],
    gpu_gpu_transfer_times: Sequence[float],
    output: Path | None,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(5, 2.3))
    x = np.arange(len(cluster_sizes), dtype=float)
    tick_stride = max(1, len(cluster_sizes) // 4)
    tick_indices = list(range(0, len(cluster_sizes), tick_stride))
    if (len(cluster_sizes) - 1) not in tick_indices:
        tick_indices.append(len(cluster_sizes) - 1)
    tick_indices = sorted(set(tick_indices))

    ax = axes[0]
    ax.plot(
        x,
        cpu_times,
        color="#5b8cc5",
        marker="o",
        linewidth=1.4,
        markersize=3.5,
        label="CPU",
    )
    ax.plot(
        x,
        gpu_times,
        color="#d46a6a",
        marker="s",
        linewidth=1.4,
        markersize=3.5,
        label="GPU",
    )
    ax.set_xticks(x[tick_indices])
    ax.set_xticklabels([str(cluster_sizes[i]) for i in tick_indices], fontsize=9)
    ax.set_ylabel("Latency (ms)", fontsize=9)
    ax.set_xlabel("Cluster size (vectors)", fontsize=9)
    ax.set_title("(a) Search Time per Cluster", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.tick_params(axis="y", labelsize=9)
    ax.tick_params(axis="x", labelsize=9)
    ax.legend(loc="upper left", fontsize=8, frameon=False)

    bar_ax = axes[1]
    bar_ax.plot(
        x,
        cpu_gpu_transfer_times,
        color="#5b8cc5",
        marker="o",
        linewidth=1.3,
        markersize=3.3,
        label="CPU→GPU transfer",
    )
    bar_ax.plot(
        x,
        gpu_gpu_transfer_times,
        color="#7dc8b1",
        marker="s",
        linewidth=1.3,
        markersize=3.3,
        label="GPU→GPU transfer",
    )
    bar_ax.plot(
        x,
        full_init_times,
        color="#d46a6a",
        marker="^",
        linewidth=1.3,
        markersize=3.3,
        label="Cluster allocation",
    )
    # if len(x) > 0:
    #     last_x = x[-1]
    #     gpu_gpu_last = gpu_gpu_transfer_times[-1] if gpu_gpu_transfer_times else float("nan")
    #     init_last = full_init_times[-1] if full_init_times else float("nan")
    #     bar_ax.annotate(
    #         f"{gpu_gpu_last:.2f} ms",
    #         xy=(last_x, gpu_gpu_last),
    #         xycoords="data",
    #         xytext=(-35, 20),
    #         textcoords="offset points",
    #         ha="left",
    #         va="top",
    #         fontsize=8,
    #         color="#7dc8b1",
    #         arrowprops=dict(arrowstyle="->", color="#7dc8b1", lw=0.8),
    #     )
    #     bar_ax.annotate(
    #         f"{init_last:.2f} ms",
    #         xy=(last_x, init_last),
    #         xycoords="data",
    #         xytext=(-40, 6),
    #         textcoords="offset points",
    #         ha="left",
    #         va="bottom",
    #         fontsize=8,
    #         color="#d46a6a",
    #         arrowprops=dict(arrowstyle="->", color="#d46a6a", lw=0.8),
    #     )
    bar_ax.set_xticks(x[tick_indices])
    bar_ax.set_xticklabels([str(cluster_sizes[i]) for i in tick_indices], fontsize=9)
    bar_ax.set_ylabel("Latency (ms)", fontsize=9)
    bar_ax.set_xlabel("Cluster size (vectors)", fontsize=9)
    bar_ax.set_title("(b) Transfer / Init Cost", fontsize=9)
    bar_ax.grid(axis="y", linestyle="--", alpha=0.3)
    bar_ax.tick_params(axis="y", labelsize=9)
    bar_ax.tick_params(axis="x", labelsize=9)
    bar_ax.legend(
        loc="upper left",
        fontsize=8,
        frameon=False,
    )

    fig.tight_layout(pad=0.45, w_pad=0.2, h_pad=0.2)
    if output:
        fig.savefig(output, dpi=300)
    else:
        plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot CPU/GPU similarity timing and dynamic GPU cache overhead for power-of-two clusters."
    )
    parser.add_argument("--min-power", type=int, default=2, help="Smallest power for 2^p cluster size. Default: 2.")
    parser.add_argument("--max-power", type=int, default=8, help="Largest power for 2^p cluster size. Default: 8.")
    parser.add_argument("--dim", type=int, default=768, help="Vector dimensionality. Default: 768.")
    parser.add_argument("--repeats", type=int, default=5, help="Timed repeats for each measurement. Default: 5.")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup runs to discard before timing. Default: 2.")
    parser.add_argument("--seed", type=int, default=123, help="RNG seed for synthetic vectors. Default: 123.")
    parser.add_argument(
        "--simulated-gpu-speedup",
        type=float,
        default=6.0,
        help="Fallback speedup factor vs CPU when CUDA is unavailable. Default: 6.0.",
    )
    parser.add_argument(
        "--pcie-gbps",
        type=float,
        default=12.0,
        help="Effective PCIe bandwidth (GB/s) for simulated transfer cost. Default: 12.0.",
    )
    parser.add_argument(
        "--full-init-overhead-ms",
        type=float,
        default=0.08,
        help="Fixed overhead (ms) for allocation-only init in simulation. Default: 0.08.",
    )
    parser.add_argument(
        "--resize-overhead-ms",
        type=float,
        default=0.12,
        help="Fixed overhead (ms) added to simulated GPU→GPU copy. Default: 0.12.",
    )
    parser.add_argument(
        "--gpu-copy-gbps",
        type=float,
        default=900.0,
        help="Effective GPU→GPU bandwidth (GB/s) when simulating device copies. Default: 900.0.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("pic/gpu_dynamic_overhead.png"),
        help="Path to save the generated figure.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if torch is None:
        print("torch not available: falling back to simulated GPU timings")
    else:
        print(f"torch {torch.__version__} import ok; cuda_available={torch.cuda.is_available()}")
    cluster_sizes = power_of_two_sizes(args.min_power, args.max_power)
    rng = np.random.default_rng(args.seed)
    (
        cpu_times,
        gpu_times,
        full_init_times,
        cpu_gpu_transfer_times,
        gpu_gpu_transfer_times,
    ) = collect_timings(cluster_sizes, args.dim, args.repeats, args.warmup, rng, args)

    plot_results(
        cluster_sizes,
        cpu_times,
        gpu_times,
        full_init_times,
        cpu_gpu_transfer_times,
        gpu_gpu_transfer_times,
        args.output,
    )


if __name__ == "__main__":
    main()
