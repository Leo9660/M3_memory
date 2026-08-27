"""
plot_bench_results.py
Read bench_results.csv written by bench_kernels_full and produce two plots:

  plot_v2_block_sweep.png  — v2 accumulate time vs block size
  plot_all_versions.png    — v1 / v2-best / v3 / v4 side-by-side bar chart
                             with per-trial scatter points
"""

import csv, os, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

CSV_PATH  = "bench_results.csv"
OUT_DIR   = "bench_plots"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load CSV ─────────────────────────────────────────────────────────────────
rows = []
with open(CSV_PATH) as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append({
            "version":   r["version"],
            "block":     int(r["block_size"]),
            "trial":     int(r["trial"]),
            "accum_ms":  float(r["accum_ms"]),
            "gatoms":    int(r["global_atomics"]),
            "notes":     r["notes"].strip(),
        })

# group → { (version, block): [accum_ms, ...] }
data = collections.defaultdict(list)
for r in rows:
    if r["version"] == "assign":
        continue
    data[(r["version"], r["block"])].append(r["accum_ms"])

def stats(vals):
    a = np.array(vals)
    return a.mean(), a.std(), a.min(), a.max()

# ─────────────────────────────────────────────────────────────────────────────
# Plot 1: v2 block-size sweep
# ─────────────────────────────────────────────────────────────────────────────
v2_blocks = sorted({b for (v, b), _ in data.items() if v == "v2"})
v2_means  = [stats(data[("v2", b)])[0] for b in v2_blocks]
v2_stds   = [stats(data[("v2", b)])[1] for b in v2_blocks]
v2_all    = [np.array(data[("v2", b)]) for b in v2_blocks]

fig, ax = plt.subplots(figsize=(7, 4.5))
x = np.arange(len(v2_blocks))
bars = ax.bar(x, v2_means, yerr=v2_stds, capsize=5,
              color="#4C9BE8", alpha=0.85, width=0.5, label="mean ± std")
# scatter individual trial points
for xi, vals in zip(x, v2_all):
    ax.scatter(np.full_like(vals, xi) + np.random.uniform(-0.12, 0.12, len(vals)),
               vals, color="#1a5fa8", s=18, alpha=0.6, zorder=3)

# annotate bars with mean value
for bar, mean in zip(bars, v2_means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f"{mean:.2f}", ha="center", va="bottom", fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels([str(b) for b in v2_blocks])
ax.set_xlabel("Block size (threads per block)", fontsize=12)
ax.set_ylabel("accumulate_v2 time (ms)", fontsize=12)
ax.set_title(f"v2 shared-mem reduction — block size sweep\n"
             f"N={65536:,}, dim=1024  |  smem per block = 8200 B (constant)", fontsize=11)
ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
ax.grid(axis="y", alpha=0.3)
ax.legend()
plt.tight_layout()
out = os.path.join(OUT_DIR, "plot_v2_block_sweep.png")
plt.savefig(out, dpi=150)
print(f"Saved {out}")

# ─────────────────────────────────────────────────────────────────────────────
# Plot 2: all-versions comparison — v1, v2-best, v3, v4
# ─────────────────────────────────────────────────────────────────────────────

# pick best v2 block
best_v2_block = min(v2_blocks, key=lambda b: stats(data[("v2", b)])[0])

versions_order = [
    ("v1",  256,          "v1\n(global\natomics)"),
    ("v2",  best_v2_block, f"v2 best\n(shared\nblock={best_v2_block})"),
    ("v3",  256,          "v3\n(warp\nshuffle)"),
    ("v4",  256,          "v4\n(scatter\n+gemv)"),
]
colors  = ["#E05C5C", "#4C9BE8", "#6BCB77", "#F4A136"]
hatches = ["", "//", "xx", ".."]

keys_present = [(v, b, lbl) for v, b, lbl in versions_order
                if (v, b) in data]

fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), gridspec_kw={"width_ratios":[2,1]})

# ── Left: absolute time bar chart ────────────────────────────────────────────
ax = axes[0]
x  = np.arange(len(keys_present))
means = [stats(data[(v, b)])[0] for v, b, _ in keys_present]
stds  = [stats(data[(v, b)])[1] for v, b, _ in keys_present]
alls  = [np.array(data[(v, b)]) for v, b, _ in keys_present]

bars2 = ax.bar(x, means, yerr=stds, capsize=6,
               color=colors[:len(keys_present)],
               hatch=[hatches[i] for i in range(len(keys_present))],
               alpha=0.85, width=0.55)
for xi, vals in zip(x, alls):
    ax.scatter(np.full_like(vals, xi) + np.random.uniform(-0.13, 0.13, len(vals)),
               vals, color="black", s=14, alpha=0.5, zorder=3)

v1_mean = stats(data[("v1", 256)])[0]
for bar, mean in zip(bars2, means):
    speedup = v1_mean / mean
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + max(stds) * 0.12,
            f"{mean:.2f} ms\n({speedup:.1f}×)", ha="center", va="bottom",
            fontsize=8.5, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels([lbl for _, _, lbl in keys_present], fontsize=9.5)
ax.set_ylabel("accumulate time (ms)", fontsize=11)
ax.set_title("Accumulate kernel: v1 vs v2-best vs v3 vs v4\n"
             f"N={65536:,}, dim=1024, BLOCK=256 (except v2-best)\nspeedup vs v1 shown",
             fontsize=10)
ax.grid(axis="y", alpha=0.3)
ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())

# ── Right: global-atomics bar chart ──────────────────────────────────────────
ax2 = axes[1]
# global atomics: v1=N*DIM, v2/v3=ceil(N/BLOCK)*DIM, v4=0
N, D, B_v2 = 65536, 1024, best_v2_block
gatom_vals = []
for v, b, _ in keys_present:
    if v == "v1":   gatom_vals.append(N * D)
    elif v == "v4": gatom_vals.append(0)
    else:           gatom_vals.append(((N + b - 1) // b) * D)

bars3 = ax2.bar(x, [g/1e6 for g in gatom_vals],
                color=colors[:len(keys_present)],
                hatch=[hatches[i] for i in range(len(keys_present))],
                alpha=0.85, width=0.55)
ax2.set_xticks(x)
ax2.set_xticklabels([lbl for _, _, lbl in keys_present], fontsize=9.5)
ax2.set_ylabel("global atomicAdds (millions)", fontsize=11)
ax2.set_title("Global atomic pressure\n(accumulate phase)", fontsize=10)
ax2.grid(axis="y", alpha=0.3)
for bar, ga in zip(bars3, gatom_vals):
    label = f"{ga/1e6:.1f}M" if ga > 0 else "0"
    ax2.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + max(gatom_vals)*0.01,
             label, ha="center", va="bottom", fontsize=9)

plt.tight_layout()
out2 = os.path.join(OUT_DIR, "plot_all_versions.png")
plt.savefig(out2, dpi=150)
print(f"Saved {out2}")

# ─────────────────────────────────────────────────────────────────────────────
# Text summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Summary ===")
print(f"{'version':20s} {'block':6s} {'mean_ms':9s} {'std_ms':8s} {'min_ms':8s} {'speedup':>8s}")
print("-" * 65)
for v, b, lbl in keys_present:
    mean, std, mn, mx = stats(data[(v, b)])
    sp = v1_mean / mean
    print(f"  {v+'@'+str(b):18s} {b:6d} {mean:9.3f} {std:8.3f} {mn:8.3f} {sp:>8.2f}x")
