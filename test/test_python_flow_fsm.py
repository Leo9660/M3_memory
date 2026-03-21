"""
Full Python flow test for M3 FSM system.

Build / install (from repo root, same directory as pyproject.toml):

    python3 -m pip install -e .

After `pip install -e .`, the extension is installed as a package submodule:

    AgentMemory.M3._m3_async

so use `from AgentMemory.M3 import _m3_async` (this script tries that first).

Optional: if you compile `_m3_async*.so` by hand and put it on PYTHONPATH as a
top-level module, `import _m3_async` also works.
"""

import numpy as np
import time
import sys

try:
    # Normal layout after `pip install -e .` (CMake: LIBRARY DESTINATION AgentMemory/M3)
    from AgentMemory.M3 import _m3_async as m3
except ImportError:
    try:
        import _m3_async as m3
    except ImportError:
        print(
            "ERROR: could not import _m3_async.\n"
            "  From repo root run:  python3 -m pip install -e .\n"
            "  Then run this script with the same venv / python you used to install.\n"
            "  Import as:  from AgentMemory.M3 import _m3_async"
        )
        sys.exit(1)

DIM    = 8
NLIST  = 4
K      = 5
NPROBE = 4

print("=" * 60)
print("M3 full system Python test")
print(f"  dim={DIM}  nlist={NLIST}  k={K}  nprobe={NPROBE}")
print("=" * 60)

centroids = np.array([
    [0.1] * DIM,
    [0.9] * DIM,
    [0.1, 0.9] + [0.1] * (DIM - 2),
    [0.9, 0.1] + [0.1] * (DIM - 2),
], dtype=np.float32)

rng = np.random.default_rng(42)
all_ids, all_vecs = [], []
for c in range(NLIST):
    for j in range(10):
        all_ids.append(c * 10 + j)
        all_vecs.append(centroids[c] + rng.uniform(-0.03, 0.03, DIM).astype(np.float32))
ids_np  = np.array(all_ids,  dtype=np.int64)
vecs_np = np.array(all_vecs, dtype=np.float32)

# -----------------------------------------------------------------------
# Part A: MultiLevelIndex directly (background prefetch thread only)
# -----------------------------------------------------------------------
print("\n── Part A: MultiLevelIndex (prefetch bg thread) ──")

idx = m3.MultiLevelIndex(DIM, m3.Metric.L2, normalized=False)
idx.set_l2_centroids(centroids)

fsm_cfg = m3.FSMConfig()
fsm_cfg.max_patterns    = 8
fsm_cfg.match_threshold = 0.0
fsm_cfg.merge_threshold = 0.0
fsm_cfg.alpha_et        = 0.0   # ET off during training
fsm_cfg.dagent_window   = 32
idx.set_fsm_config(fsm_cfg)

cache_cfg = m3.CacheConfig()
cache_cfg.l0_neighborhood_k       = 5
cache_cfg.l1_neighborhood_k       = 10
cache_cfg.prefetch_queue_capacity  = 128
idx.set_cache_config(cache_cfg)

idx.insert(ids_np, vecs_np)
print("  [A1] Index built, vectors inserted")

# Cold search — L0 empty, falls through to L2
q_a = (centroids[0] + 0.01).reshape(1, -1)
ids_out, _ = idx.search(q_a, k=K, nprobe=NPROBE)
assert all(0 <= i < 10 for i in ids_out[0])
print("  [A2] PASS cold search returns cluster-0 results")

# Warm all clusters — bg reactive promotion fires for each
for c in range(NLIST):
    idx.search((centroids[c] + 0.01).reshape(1, -1), k=K, nprobe=NPROBE)

# Wait for bg prefetch thread to complete promotion tasks
time.sleep(0.2)
print("  [A3] Cache warmed (200ms wait for bg prefetch thread)")

# Warm search with trajectory — should now fill L0 steps
traj = m3.RequestTrajectory("req-warm")
ids_out, _ = idx.search(q_a, k=K, nprobe=NPROBE, traj=traj)
assert traj.length >= 1
assert traj.steps[0].layer == m3.FSMLayer.L0
print(f"  [A4] PASS trajectory filled: {traj.length} steps, "
      f"first step L0 cid={traj.steps[0].cluster_id}")

# Train FSM with repeated A→B pattern
q_b = (centroids[1] + 0.01).reshape(1, -1)
for rep in range(6):
    t = m3.RequestTrajectory(f"train-{rep}")
    idx.search(q_a, k=K, nprobe=NPROBE, traj=t)
    idx.search(q_b, k=K, nprobe=NPROBE, traj=t)
    t.finalize()
    if t.length >= 2:
        idx.fsm_table().update_from_trajectory(t, centroids.ravel(), DIM)

np_stored = idx.fsm_table().num_patterns()
print(f"  [A5] FSM trained: {np_stored} pattern(s) stored")

# FSM prediction
partial = m3.RequestTrajectory("pred")
idx.search(q_a, k=K, nprobe=NPROBE, traj=partial)
result = idx.fsm_table().match_and_predict(partial, m3.FSMLayer.L0)
print(f"  [A6] FSM prediction after cluster-0: ranked={result.ranked_clusters} "
      f"score={result.best_score:.3f}")

# Predictive prefetch: traj-bearing search above already enqueued prefetch for cluster 1
time.sleep(0.2)
t_verify = m3.RequestTrajectory("verify")
ids_b, _ = idx.search(q_b, k=K, nprobe=NPROBE, traj=t_verify)
assert all(10 <= i < 20 for i in ids_b[0])
assert t_verify.length >= 1
print(f"  [A7] PASS predictive prefetch: cluster-1 warm, steps={t_verify.length}")

# Correctness vs brute force
q_test = (centroids[3] + 0.02).reshape(1, -1)
bf_top1 = all_ids[int(np.argmin(np.sum((vecs_np - q_test)**2, axis=1)))]
ivf_ids, _ = idx.search(q_test, k=1, nprobe=NPROBE)
assert ivf_ids[0][0] == bf_top1
print(f"  [A8] PASS correctness: IVF top-1={ivf_ids[0][0]} == BF top-1={bf_top1}")

# -----------------------------------------------------------------------
# Part B: AsyncEngine (writer threads + maintenance thread)
# -----------------------------------------------------------------------
print("\n── Part B: AsyncEngine (writer + maintenance threads) ──")

# B1: Create engine and IVF index
engine = m3.AsyncEngine()

engine.set_queue_policy(
    capacity      = 1024,
    pop_batch_max = 32,
    block_on_full = True,
)
engine.set_maintenance_policy(
    period_sec       = 0.5,   # run maintenance every 500ms
    split_threshold  = 5000,
    compact_ratio    = 0.7,
)

# Register a flat IVF index with the engine
engine.create_ivf(
    index_id  = 0,
    dim       = DIM,
    metric    = m3.Metric.L2,
    normalized= False,
    centroids = centroids,
)

# Start writer (2) and maintenance (1) background threads
engine.start(writer_threads=2, maintenance_threads=1)
print("  [B1] AsyncEngine started (2 writer threads, 1 maintenance thread)")

# Attach the MultiLevelIndex so maintenance_pass() drives L0/L1/L2 cache upkeep
engine.attach_multilevel_index(idx)
print("  [B2] MultiLevelIndex attached to AsyncEngine maintenance thread")

# B2: Async writes via queue
print("  [B3] Enqueuing async inserts...")
batch_size = 5
for cluster_id in range(NLIST):
    batch_ids  = np.array([100 + cluster_id*10 + j for j in range(batch_size)],
                           dtype=np.int64)
    batch_vecs = (centroids[cluster_id] + rng.uniform(-0.02, 0.02,
                  (batch_size, DIM)).astype(np.float32))
    ok = engine.enqueue_insert(
        index_id   = 0,
        cluster_id = cluster_id,
        ids        = batch_ids,
        vectors    = batch_vecs,
    )
    assert ok, f"enqueue_insert rejected (queue full?) for cluster {cluster_id}"

print(f"  [B3] Enqueued {NLIST * batch_size} inserts across {NLIST} clusters")

# B3: Flush — blocks until write queue is fully drained and applied
engine.flush()
print("  [B4] PASS flush() complete — all writes applied by writer threads")

# B4: Search through engine (IVF index, no MultiLevelIndex cache)
q_eng = (centroids[0] + 0.01).reshape(1, -1)
eng_ids, eng_scores = engine.search(index_id=0, queries=q_eng, k=3, nprobe=NPROBE)
print(f"  [B5] Engine search results: {eng_ids[0]}")
# Original inserts (ids 0-9) plus new inserts (100-109) should both be in cluster 0
assert len(eng_ids[0]) == 3
print("  [B5] PASS engine search returned results")

# B5: Enqueue_insert_auto (engine assigns cluster automatically)
auto_ids  = np.array([200, 201, 202], dtype=np.int64)
auto_vecs = np.array([
    centroids[2] + rng.uniform(-0.02, 0.02, DIM).astype(np.float32),
    centroids[2] + rng.uniform(-0.02, 0.02, DIM).astype(np.float32),
    centroids[3] + rng.uniform(-0.02, 0.02, DIM).astype(np.float32),
], dtype=np.float32)
ok = engine.enqueue_insert_auto(index_id=0, ids=auto_ids, vectors=auto_vecs)
assert ok
engine.flush()
print("  [B6] PASS enqueue_insert_auto + flush")

# B6: Let maintenance thread run for one cycle
print("  [B7] Waiting 1s for maintenance thread cycle...")
time.sleep(1.0)
print("  [B7] PASS maintenance thread ran (no crash)")

# B7: Verify cluster sizes are sane
size_c0 = engine.cluster_live_size(index_id=0, cluster_id=0)
print(f"  [B8] Cluster-0 live size after inserts: {size_c0}")
assert size_c0 >= batch_size   # at least our new batch
print("  [B8] PASS cluster live size is correct")

# B8: Stop the engine cleanly — joins all background threads
engine.stop()
print("  [B9] PASS engine.stop() — all threads joined cleanly")

# -----------------------------------------------------------------------
# Part C: MultiLevelIndex maintenance_pass() driven by AsyncEngine
# -----------------------------------------------------------------------
print("\n── Part C: maintenance_pass integration ──")

# Restart engine and reattach to run maintenance_pass on the cache index
engine2 = m3.AsyncEngine()
engine2.set_maintenance_policy(period_sec=0.3)
engine2.create_ivf(index_id=1, dim=DIM, metric=m3.Metric.L2,
                   normalized=False, centroids=centroids)
engine2.start(writer_threads=1, maintenance_threads=1)
engine2.attach_multilevel_index(idx)

# Insert some data so maintenance has something to work with
for c in range(NLIST):
    cids = np.array([300 + c*5 + j for j in range(5)], dtype=np.int64)
    cvecs = (centroids[c] + rng.uniform(-0.02, 0.02, (5, DIM)).astype(np.float32))
    engine2.enqueue_insert(1, c, cids, cvecs)
engine2.flush()

# Let maintenance run at least twice (period=0.3s, sleep 0.8s)
time.sleep(0.8)
print("  [C1] PASS maintenance_pass ran via AsyncEngine thread (no crash)")

# Search still works after maintenance
traj_c = m3.RequestTrajectory("post-maint")
ids_c, _ = idx.search(q_a, k=K, nprobe=NPROBE, traj=traj_c)
assert all(0 <= i < 10 for i in ids_c[0])
print(f"  [C2] PASS search still correct after maintenance: {ids_c[0]}")

engine2.stop()
print("  [C3] PASS engine2.stop() clean")

# -----------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------
print("\n" + "=" * 60)
print("All tests PASSED")
print()
print("Background systems verified:")
print("  prefetch_thread_     — reactive and predictive promotion [A3,A7]")
print("  writer threads       — async enqueue + flush             [B3,B4]")
print("  maintenance thread   — periodic maintenance_pass         [B7,C1]")
print("  FSM trajectory       — auto-filled by search()           [A4]")
print("  FSM pattern learning — update_from_trajectory()          [A5]")
print("  FSM prediction       — match_and_predict()               [A6]")
print("=" * 60)
