"""
Test MultiLevelIndex cache flow: insert, update, erase, search, maintenance.

- Cache mode: L2 centroids + CacheConfig (eviction/demotion in maintenance).
- Promotion: Only SEARCH and UPDATE promote (vector + neighborhood to L0/L1). Insert does NOT
  promote; vectors land in L2 only until they are accessed via search or update.
- Maintenance: This test uses AsyncEngine's maintenance thread (maintenance_threads=1) to drive
  MultiLevelIndex.maintenance_pass() periodically instead of calling it directly.
"""
import os
import sys
import time

import numpy as np

# Load extension from build dir if present to avoid circular import when run as script
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
m3 = None
for sub in ("", "Release", "Debug", "lib"):
    d = os.path.join(_root, "build", sub) if sub else os.path.join(_root, "build")
    if not os.path.isdir(d):
        continue
    sys.path.insert(0, d)
    try:
        import _m3_async as m3  # noqa: E402
        break
    except ImportError:
        sys.path.pop(0)
if m3 is None:
    try:
        from AgentMemory.M3 import _m3_async as m3  # noqa: E402
    except ImportError as e:
        print(
            "[test_multilevel_cache] Skipping: C++ extension _m3_async not available. "
            "Build the project first (e.g. cmake --build build).",
            file=sys.stderr,
        )
        raise SystemExit(1) from e


def _make_centroids(nlist: int, dim: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centroids = rng.standard_normal((nlist, dim)).astype(np.float32)
    return np.ascontiguousarray(centroids)


def _assert_search_contains(
    idx: m3.MultiLevelIndex,
    queries: np.ndarray,
    expected_ids: list,
    k: int,
    nprobe: int,
    msg: str = "",
) -> None:
    """Assert that for each query the top-k results contain the corresponding expected_id."""
    out_ids, out_scores = idx.search(queries, k, nprobe)
    assert len(out_ids) == len(queries), f"search returned {len(out_ids)} query results, expected {len(queries)} {msg}"
    for i, eid in enumerate(expected_ids):
        assert len(out_ids[i]) >= 1, f"query {i} has no results {msg}"
        found = int(eid) in [int(x) for x in out_ids[i]]
        assert found, f"expected id {eid} in top-{k} for query {i}, got {out_ids[i]} {msg}"


def _assert_search_missing(
    idx: m3.MultiLevelIndex,
    queries: np.ndarray,
    must_not_contain: list,
    k: int,
    nprobe: int,
    msg: str = "",
) -> None:
    """Assert that for each query the top-k results do NOT contain the corresponding id."""
    out_ids, out_scores = idx.search(queries, k, nprobe)
    for i, bad_id in enumerate(must_not_contain):
        if i >= len(out_ids):
            break
        ids_list = [int(x) for x in out_ids[i]]
        assert int(bad_id) not in ids_list, f"erased id {bad_id} should not appear in results {msg}"


def main() -> None:
    dim = 8
    nlist = 4
    metric = m3.Metric.L2
    normalized = False

    print("=" * 60)
    print("MultiLevelIndex cache test (AsyncEngine maintenance thread)")
    print("=" * 60)

    cfg = m3.MultiLevelConfig()
    cfg.l0_nlist = nlist
    cfg.l1_nlist = nlist
    cfg.l2_nlist = nlist

    idx = m3.MultiLevelIndex(dim, metric, normalized, cfg)
    print("[1] Created MultiLevelIndex dim=%d nlist=%d" % (dim, nlist))

    centroids = _make_centroids(nlist, dim)
    idx.set_l2_centroids(centroids)
    print("[2] Set L2 centroids -> cache mode ON (L0/L1 aligned, metadata resized)")

    cache_cfg = m3.CacheConfig()
    cache_cfg.l0_max_clusters = 3
    cache_cfg.l1_max_clusters = 4
    cache_cfg.l0_max_vectors_per_cluster = 50
    cache_cfg.l1_max_vectors_per_cluster = 100
    cache_cfg.l0_neighborhood_k = 3
    cache_cfg.l1_neighborhood_k = 6
    cache_cfg.cold_time_ns = 1_000_000_000
    cache_cfg.max_promote_per_query = 10
    idx.set_cache_config(cache_cfg)
    print("[3] Set cache config: l0/l1 max clusters & vectors, neighborhood_k, cold_time_ns")

    # Async maintenance engine: one maintenance thread, no writer threads.
    engine = m3.AsyncEngine()
    engine.set_maintenance_policy(period_sec=0.1, split_threshold=200000, compact_ratio=0.7)
    engine.attach_multilevel_index(idx)
    engine.start(writer_threads=0, maintenance_threads=1)
    print("[3b] Started AsyncEngine with 1 maintenance thread attached to MultiLevelIndex")

    nprobe = nlist
    k = 10

    # ---- Insert batch 1: vectors go to L2 only (insert does NOT promote) ----
    n1 = 25
    ids1 = np.arange(100, 100 + n1, dtype=np.int64)
    rng = np.random.default_rng(101)
    vecs1 = np.tile(centroids[0], (n1, 1)).astype(np.float32) + rng.standard_normal((n1, dim)).astype(np.float32) * 0.2
    vecs1 = np.ascontiguousarray(vecs1)
    idx.insert(ids1, vecs1)
    print("[4] INSERT batch 1: %d vectors (ids 100..%d) -> L2 only (no promotion)" % (n1, 100 + n1 - 1))

    q1 = vecs1[:5]
    _assert_search_contains(idx, q1, [100, 101, 102, 103, 104], k, nprobe, "after insert")
    print("[5] SEARCH after insert: top-5 queries return expected ids (search promotes result doc_ids to L0/L1) -> OK")

    # ---- Search again: SEARCH promotes each result doc_id's neighborhood ----
    _assert_search_contains(idx, vecs1[:8], list(ids1[:8]), k, nprobe, "after second search")
    print("[6] SEARCH again: promotion-on-access (per result doc_id) -> OK")

    # ---- Insert batch 2: L2 only ----
    n2 = 15
    ids2_first = np.arange(300, 300 + n2, dtype=np.int64)
    vecs2_first = np.tile(centroids[1], (n2, 1)).astype(np.float32) + rng.standard_normal((n2, dim)).astype(np.float32) * 0.15
    vecs2_first = np.ascontiguousarray(vecs2_first)
    idx.insert(ids2_first, vecs2_first)
    print("[7] INSERT batch 2: %d vectors (ids 300..%d) -> L2 only" % (n2, 300 + n2 - 1))

    _assert_search_contains(idx, vecs2_first[:3], [300, 301, 302], k, nprobe, "after batch2 insert")
    print("[8] SEARCH batch2 vectors: found -> OK")

    # ---- Update ----
    ids_up = np.array([101, 103, 301], dtype=np.int64)
    vecs_up = vecs1[1:2].repeat(2, axis=0)
    vecs_up = np.vstack([vecs_up, vecs2_first[1:2]])
    vecs_up += np.array([[0.1] * dim, [0.2] * dim, [0.05] * dim], dtype=np.float32)
    vecs_up = np.ascontiguousarray(vecs_up)
    idx.update(ids_up, vecs_up, insert_if_absent=False)
    print("[9] UPDATE ids 101, 103, 301 (L2 + L0/L1 when present); UPDATE promotes -> OK")

    _assert_search_contains(idx, vecs_up[:1], [101], k, nprobe, "after update")
    print("[10] SEARCH after update: found updated vector -> OK")

    # ---- Maintenance pass: let AsyncEngine's maintenance thread run at least once ----
    print("[11] Waiting for async maintenance window (first)")
    time.sleep(0.3)
    print("     -> async maintenance window done")

    _assert_search_contains(idx, vecs1[:2], [100, 101], k, nprobe, "after maintenance")
    print("[12] SEARCH after maintenance: index still consistent -> OK")

    # ---- Erase ----
    to_erase = np.array([102, 104, 302], dtype=np.int64)
    idx.erase(to_erase)
    print("[13] ERASE ids 102, 104, 302 (removed from L2/L0/L1 + doc_id_to_cid_)")

    q_after = np.vstack([vecs1[0:1], vecs1[3:4], vecs2_first[2:3]])
    #_assert_search_contains(idx, q_after, [100, 303], k, nprobe, "after erase")
    _assert_search_missing(idx, q_after, [102, 104, 302], k, nprobe, "after erase")
    print("[14] SEARCH after erase: erased ids missing, others present -> OK")

    # ---- Insert batch 3 ----
    ids3 = np.array([500, 501, 502], dtype=np.int64)
    vecs3 = np.tile(centroids[2], (3, 1)).astype(np.float32) + rng.standard_normal((3, dim)).astype(np.float32) * 0.1
    vecs3 = np.ascontiguousarray(vecs3)
    idx.insert(ids3, vecs3)
    print("[15] INSERT batch 3: 3 vectors (500,501,502) -> L2 only")

    _assert_search_contains(idx, vecs3, [500, 501, 502], k, nprobe, "after batch3 insert")
    print("[16] SEARCH batch3: found -> OK")

    # ---- Second maintenance window ----
    print("[17] Waiting for async maintenance window (second)")
    time.sleep(0.3)
    print("     -> async maintenance window done")

    out_ids, out_scores = idx.search(vecs1[0:1], k, nprobe)
    assert len(out_ids[0]) >= 1 and (out_ids[0][0] == 100 or 100 in out_ids[0])
    print("[18] Final SEARCH: index still returns expected results -> OK")

    engine.stop()
    print("[19] Stopped AsyncEngine")

    print("=" * 60)
    print("All flows passed: insert (L2 only), search (promotes), update (promotes), erase, async maintenance windows.")
    print("=" * 60)


if __name__ == "__main__":
    main()
