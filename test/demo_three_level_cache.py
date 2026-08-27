"""
Annotated demo of the Three-Level Cluster Cache (L0 / L1 / L2).

This script walks through each stage of the cache lifecycle step by step,
printing what is happening at the Python level. Set M3_DEBUG=1 to also see
detailed C++ logs for every search stage, promotion, writeback, and merge.

Usage:
    python test/demo_three_level_cache.py          # Python-level annotations only
    M3_DEBUG=1 python test/demo_three_level_cache.py  # + C++ per-event logs on stderr
"""
import os
import sys
import time

import numpy as np

# ---------------------------------------------------------------------------
# Resolve C++ extension
# ---------------------------------------------------------------------------
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
m3 = None
for sub in ("", "Release", "Debug", "lib"):
    d = os.path.join(_root, "build", sub) if sub else os.path.join(_root, "build")
    if not os.path.isdir(d):
        continue
    sys.path.insert(0, d)
    try:
        import _m3_async as m3
        break
    except ImportError:
        sys.path.pop(0)
if m3 is None:
    try:
        from AgentMemory.M3 import _m3_async as m3
    except ImportError as e:
        print("Build the project first (cmake --build build).", file=sys.stderr)
        raise SystemExit(1) from e

VERBOSE = os.environ.get("M3_DEBUG") is not None

# ---------------------------------------------------------------------------
# Parameters — deliberately small to trigger overflow quickly
# ---------------------------------------------------------------------------
DIM           = 3        # embedding dimension
NLIST         = 3        # number of IVF clusters at every level
K             = 3        # retrieval parameter k
K_PRIME_L1    = 5        # k' — neighbourhood size promoted into L1 (wider than k)
                         # L0 stores only the directly accessed vector (no neighbourhood, per spec)
L0_CAP        = 5        # max vectors per L0 cluster  → overflow with ≥7
L1_CAP        = 8       # max vectors per L1 cluster  → overflow with ≥13
ALPHA_ET      = 0.7      # early-termination factor αet
DAGENT_WINDOW = 5        # rolling-average window for dagent


def sep(title=""):
    width = 70
    if title:
        pad = (width - len(title) - 2) // 2
        print("\n" + "─" * pad + f" {title} " + "─" * (width - pad - len(title) - 2))
    else:
        print("\n" + "─" * width)


def note(msg):
    print(f"  ▶ {msg}")


def stat(label, value):
    print(f"    {label:<40} {value}")


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
sep("SETUP")
note("Creating MultiLevelIndex (L0 / L1 / L2) in cache mode")
note(f"  dim={DIM}  nlist={NLIST}  k={K}  k'(L1)={K_PRIME_L1}  (L0 stores accessed vec only)")
note(f"  L0 cap={L0_CAP} vecs/cluster   L1 cap={L1_CAP} vecs/cluster")
note(f"  αet={ALPHA_ET}  dagent_window={DAGENT_WINDOW}")

cfg = m3.MultiLevelConfig()
cfg.l0_nlist = NLIST
cfg.l1_nlist = NLIST
cfg.l2_nlist = NLIST
idx = m3.MultiLevelIndex(DIM, m3.Metric.L2, False, cfg)

rng = np.random.default_rng(0)
centroids = rng.standard_normal((NLIST, DIM)).astype(np.float32)
centroids = np.ascontiguousarray(centroids)

# set_l2_centroids aligns L0/L1 to the same centroid grid and enables cache mode
idx.set_l2_centroids(centroids)
note("set_l2_centroids() → cache mode ENABLED (L0/L1 aligned to L2 grid, metadata populated)")

cc = m3.CacheConfig()
cc.l0_max_clusters           = NLIST
cc.l1_max_clusters           = NLIST
cc.l0_max_vectors_per_cluster = L0_CAP
cc.l1_max_vectors_per_cluster = L1_CAP
cc.l1_neighborhood_k         = K_PRIME_L1
cc.cold_time_ns              = 120_000_000_000   # 120 s — cold demotion disabled for demo
cc.alpha_et                  = ALPHA_ET
cc.dagent_window             = DAGENT_WINDOW
idx.set_cache_config(cc)

# Optionally attach an AsyncEngine so maintenance runs in the background.
# For the demo we call maintenance_pass() manually to make timing predictable.


# ---------------------------------------------------------------------------
# Phase 1: INSERT — vectors land in L2 only
# ---------------------------------------------------------------------------
sep("PHASE 1 — INSERT (vectors land in L2 only; no cache promotion on insert)")

N_PHASE1 = 10
ids_p1 = np.arange(100, 100 + N_PHASE1, dtype=np.int64)
vecs_p1 = (
    np.tile(centroids[0], (N_PHASE1, 1)).astype(np.float32)
    + rng.standard_normal((N_PHASE1, DIM)).astype(np.float32) * 0.15
)
vecs_p1 = np.ascontiguousarray(vecs_p1)

idx.insert(ids_p1, vecs_p1)
note(f"Inserted {N_PHASE1} vectors (ids 100…{100+N_PHASE1-1}) near centroid[0]")
stat("L2 cluster 0 (canonical store)", f"~{N_PHASE1} vectors")
stat("L0 cluster 0", "0 vectors  (insert does NOT promote)")
stat("L1 cluster 0", "0 vectors  (insert does NOT promote)")


# ---------------------------------------------------------------------------
# Phase 2: SEARCH — promotes results into L0 and L1 (k')
# ---------------------------------------------------------------------------
sep("PHASE 2 — SEARCH (promotes accessed vectors into L0, into L1 with k')")

note(f"Querying first 3 vectors; k={K}, nprobe={NLIST}")
note(f"  Per hit, the accessed vector itself → L0  (temporal locality, no neighbourhood search)")
note(f"  Per hit, promote k'={K_PRIME_L1} nearest neighbours → L1 (wider neighbourhood)")
if VERBOSE:
    note("  [M3_DEBUG] C++ will print [M3:search] and [M3:promote] lines below ↓")

q1 = vecs_p1[:3]
out_ids1, out_scores1 = idx.search(q1, k=K, nprobe=NLIST)

for qi, (ids_row, scores_row) in enumerate(zip(out_ids1, out_scores1)):
    print(f"    query {qi}: top-{K} ids = {[int(x) for x in ids_row]}")
    print(f"             scores = {[round(float(s), 4) for s in scores_row]}")

note("After search: L0 and L1 now hold the promoted neighbourhood around each hit.")


# ---------------------------------------------------------------------------
# Phase 3: MORE INSERTS + SEARCHES — build up dagent
# ---------------------------------------------------------------------------
sep("PHASE 3 — MORE QUERIES to warm up dagent rolling average")

N_WARMUP = DAGENT_WINDOW + 2
note(f"Running {N_WARMUP} additional queries (need ≥{DAGENT_WINDOW} to fill dagent window).")
note(f"  After {DAGENT_WINDOW} queries: dynamic threshold = αet({ALPHA_ET}) × dagent(mean top-k dist)")
if VERBOSE:
    note("  [M3_DEBUG] Watch [M3:dagent] lines to see dagent evolve ↓")

for i in range(N_WARMUP):
    idx.search(vecs_p1[i % len(vecs_p1) : i % len(vecs_p1) + 1], k=K, nprobe=NLIST)
    if i == DAGENT_WINDOW - 1:
        note(f"  → dagent window now full after query {i+1} — dynamic threshold is LIVE")


# # ---------------------------------------------------------------------------
# # Phase 4: CLOSE QUERY — should trigger early exit at L0 or L1
# # ---------------------------------------------------------------------------
# sep("PHASE 4 — CLOSE QUERY (dynamic αet·dagent should cause early-exit)")

# note("Querying with the same vectors already promoted to L0:")
# note(f"  If kth-score ≤ αet({ALPHA_ET}) × dagent, search stops at L0 or L1 — no L2 scan.")
# if VERBOSE:
#     note("  Watch [M3:search] stage column: 'L0-only' or 'L0+L1 (early-exit)' means threshold fired.")

# out_ids_et, out_scores_et = idx.search(vecs_p1[:5], k=K, nprobe=NLIST)
# for qi, (ids_row, scores_row) in enumerate(zip(out_ids_et, out_scores_et)):
#     kth = float(scores_row[K-1]) if len(scores_row) >= K else float(scores_row[-1]) if scores_row else float("inf")
#     print(f"    query {qi}: kth-score={kth:.4f}  results={[int(x) for x in ids_row]}")


# ---------------------------------------------------------------------------
# Phase 5: OVERFLOW L0 — trigger L0→L1 writeback
# ---------------------------------------------------------------------------
sep("PHASE 5 — OVERFLOW L0 → L1 writeback during maintenance_pass()")

note(f"L0 cap per cluster = {L0_CAP} vectors.")
note(f"Each search promoted the accessed vector into L0 (cap={L0_CAP}).")
note("Inserting more vectors and searching to push L0 well past its cap.")

N_OVERFLOW = L0_CAP * 3   # definitely overflow
ids_ov = np.arange(200, 200 + N_OVERFLOW, dtype=np.int64)
vecs_ov = (
    np.tile(centroids[0], (N_OVERFLOW, 1)).astype(np.float32)
    + rng.standard_normal((N_OVERFLOW, DIM)).astype(np.float32) * 0.1
)
vecs_ov = np.ascontiguousarray(vecs_ov)
idx.insert(ids_ov, vecs_ov)
# Search to promote them into L0
idx.search(vecs_ov, k=K, nprobe=NLIST)
note(f"Inserted {N_OVERFLOW} more vectors and searched → L0 cluster 0 is now well over cap {L0_CAP}.")

note("Calling maintenance_pass() — L0 overflow vectors are written back to L1, THEN erased from L0.")
if VERBOSE:
    note("  Watch [M3:maint] L0->L1 writeback lines ↓")

idx.maintenance_pass()
note("maintenance_pass() done.")

# Verify all vectors still findable (they landed in L1/L2)
all_ids = np.concatenate([ids_p1, ids_ov])
all_vecs = np.vstack([vecs_p1, vecs_ov])
out_after_maint, _ = idx.search(all_vecs, k=K, nprobe=NLIST)
found = {int(x) for row in out_after_maint for x in row}
missing = [int(i) for i in all_ids if int(i) not in found]
if missing:
    note(f"WARNING: {len(missing)} vectors missing after L0→L1 writeback: {missing[:5]}")
else:
    note(f"All {len(all_ids)} vectors still reachable after L0→L1 writeback. ✓")


# ---------------------------------------------------------------------------
# Phase 6: COLD-CLUSTER DEMOTION
# ---------------------------------------------------------------------------
sep("PHASE 6 — COLD-CLUSTER DEMOTION during maintenance_pass()")

note("Populate clusters 1 and 2 by inserting+searching vectors near centroids[1] and [2].")
ids_c1 = np.arange(300, 310, dtype=np.int64)
vecs_c1 = (
    np.tile(centroids[1], (10, 1)).astype(np.float32)
    + rng.standard_normal((10, DIM)).astype(np.float32) * 0.1
)
vecs_c1 = np.ascontiguousarray(vecs_c1)
ids_c2 = np.arange(410, 420, dtype=np.int64)
vecs_c2 = (
    np.tile(centroids[2], (10, 1)).astype(np.float32)
    + rng.standard_normal((10, DIM)).astype(np.float32) * 0.1
)
vecs_c2 = np.ascontiguousarray(vecs_c2)
idx.insert(ids_c1, vecs_c1)
idx.insert(ids_c2, vecs_c2)
idx.search(vecs_c1, k=K, nprobe=NLIST)   # promotes cluster 1 into L0/L1
idx.search(vecs_c2, k=K, nprobe=NLIST)   # promotes cluster 2 into L0/L1
note("Clusters 1 and 2 are now active in L0/L1.")

# Set cold_time_ns = 10 ms so all 3 clusters go cold quickly
COLD_NS = 10_000_000   # 10 ms
cc2 = m3.CacheConfig()
cc2.l0_max_clusters            = NLIST
cc2.l1_max_clusters            = NLIST
cc2.l0_max_vectors_per_cluster = L0_CAP
cc2.l1_max_vectors_per_cluster = L1_CAP
cc2.l1_neighborhood_k          = K_PRIME_L1
cc2.cold_time_ns               = COLD_NS
cc2.alpha_et                   = ALPHA_ET
cc2.dagent_window              = DAGENT_WINDOW
idx.set_cache_config(cc2)

note(f"Set cold_time_ns={COLD_NS/1e6:.0f} ms — sleeping 50 ms so all clusters go cold...")
time.sleep(0.05)

note("Calling maintenance_pass() — cold clusters demoted from L0/L1 (still in L2).")
if VERBOSE:
    note("  Watch [M3:maint] cold-cluster demotion lines ↓")
idx.maintenance_pass()
note("maintenance_pass() done.")

out_c1, _ = idx.search(vecs_c1[:3], k=K, nprobe=NLIST)
found_c1 = {int(x) for row in out_c1 for x in row}
note(f"cluster-1 vectors still reachable via L2 after cold demotion: "
     f"{len(found_c1 & set(int(i) for i in ids_c1))} / 10 found. ✓")


# ---------------------------------------------------------------------------
# Phase 7: CLUSTER-COUNT DEMOTION
# ---------------------------------------------------------------------------
sep("PHASE 7 — CLUSTER-COUNT DEMOTION during maintenance_pass()")

note("Setting l0_max_clusters=1 and l1_max_clusters=1 (currently 3 clusters could be active).")
note("Re-touching all 3 clusters via search, then calling maintenance_pass().")
note("→ The 2 LRU clusters will be evicted from L0/L1; L2 is unaffected.")

cc3 = m3.CacheConfig()
cc3.l0_max_clusters            = 1
cc3.l1_max_clusters            = 1
cc3.l0_max_vectors_per_cluster = L0_CAP
cc3.l1_max_vectors_per_cluster = L1_CAP
cc3.l1_neighborhood_k          = K_PRIME_L1
cc3.cold_time_ns               = 120_000_000_000   # 120s — disable cold demotion
cc3.alpha_et                   = ALPHA_ET
cc3.dagent_window              = DAGENT_WINDOW
idx.set_cache_config(cc3)

# Touch clusters in order: 0 first (oldest), 2 last (most recent)
idx.search(vecs_p1[:1], k=K, nprobe=NLIST)   # cluster 0 — touched first (LRU)
idx.search(vecs_c1[:1], k=K, nprobe=NLIST)   # cluster 1
idx.search(vecs_c2[:1], k=K, nprobe=NLIST)   # cluster 2 — touched last (MRU)
note("3 clusters touched (0=oldest, 2=most-recent). max_clusters=1 → 2 will be evicted.")

if VERBOSE:
    note("  Watch [M3:maint] cluster-count check and cluster-count demotion lines ↓")
idx.maintenance_pass()
note("maintenance_pass() done. Clusters 0 and 1 evicted from L0/L1; cluster 2 retained.")
note("All vectors still reachable via L2 (canonical store is unaffected by cache demotion).")


# ---------------------------------------------------------------------------
# Phase 8: ERASE
# ---------------------------------------------------------------------------
sep("PHASE 8 — ERASE (removed from all levels)")

# reset caps back so search works normally for erase verification
idx.set_cache_config(cc2)   # restore permissive cluster caps

to_erase = np.array([100, 101, 200, 201], dtype=np.int64)
idx.erase(to_erase)
note(f"Erased ids {list(to_erase)} from L0, L1, L2 and doc_id_to_cid_ map.")

out_er, _ = idx.search(vecs_p1[:4], k=K, nprobe=NLIST)
erased_found = [i for i in to_erase if int(i) in {int(x) for row in out_er for x in row}]
if erased_found:
    note(f"WARNING: erased ids still found: {erased_found}")
else:
    note("Erased ids are absent from search results. ✓")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
# sep("SUMMARY")
# print("""
#   What you just observed:
#   ─────────────────────────────────────────────────────────────────────
#   INSERT       → vectors go to L2 only (canonical store). No promotion.

#   SEARCH       → each result doc_id triggers promotion:
#                    • k  nearest in L2 cluster  → copied into L0 (tiny, hot)
#                    • k' nearest in L2 cluster  → copied into L1 (intermediate)
#                  (k' > k so L1 holds a broader neighbourhood than L0)

#   Early-exit   → once dagent is warm, if all top-k scores ≤ αet·dagent
#                  the search returns from L0 (or L0+L1) without scanning L2.

#   maintenance_pass() step 3:
#     L0 overflow → evicted vectors are written back into L1 (not dropped).
#     L1 overflow → evicted vectors are merged into L2 (not dropped).

#   ERASE        → removed from every level.
#   ─────────────────────────────────────────────────────────────────────
#   Re-run with:  M3_DEBUG=1 python test/demo_three_level_cache.py
#   to see C++ per-event logs for every search stage, promotion, writeback,
#   merge, and dagent update.
# """)
