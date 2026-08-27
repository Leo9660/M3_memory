import numpy as np

from AgentMemory.M3 import _m3_async as m3


def _assert_search_finds_self(eng: "m3.AsyncEngine", index_id: int, vecs: np.ndarray, ids: np.ndarray, nprobe: int):
    # For each query vec, ensure its own id is returned at rank-1 when searching across all clusters.
    for i in range(vecs.shape[0]):
        q = vecs[i : i + 1].astype(np.float32, copy=False)
        out_ids, out_scores = eng.search(index_id, q, 1, nprobe)
        assert len(out_ids) == 1 and len(out_ids[0]) == 1
        assert out_ids[0][0] == int(ids[i])


def main():
    eng = m3.AsyncEngine()
    # No need for maintenance threads; we call split/merge directly.
    eng.start(writer_threads=1, maintenance_threads=0)
    print("[split_merge] AsyncEngine started")

    index_id = 0
    dim = 16
    metric = m3.Metric.L2
    normalized = False

    # Start with a single cluster (cluster_id=0).
    centroids = np.zeros((1, dim), dtype=np.float32)
    eng.create_ivf(index_id, dim, metric, normalized, centroids)
    print(f"[split_merge] Created IVF index {index_id} with dim={dim}, nlist=1")

    # Insert two separable blobs so k-means(2) split is stable.
    n0, n1 = 40, 40
    a = np.random.randn(n0, dim).astype(np.float32) * 0.01 + 0.0
    b = np.random.randn(n1, dim).astype(np.float32) * 0.01 + 10.0
    vecs = np.vstack([a, b]).astype(np.float32, copy=False)
    ids = np.arange(1000, 1000 + vecs.shape[0], dtype=np.int64)

    ok = eng.enqueue_insert(index_id, 0, ids, vecs)
    assert ok
    eng.flush()

    assert eng.nlist_of(index_id) == 1
    assert eng.cluster_live_size(index_id, 0) == vecs.shape[0]
    assert eng.cluster_valid(index_id, 0) is True
    print(f"[split_merge] Inserted {vecs.shape[0]} vectors into cluster 0")

    # Force split with a small threshold.
    new_cid = eng.split_cluster(index_id, 0, max_vectors_before_split=10)
    assert new_cid >= 0
    assert eng.nlist_of(index_id) == 2
    assert eng.cluster_valid(index_id, 0) is True
    assert eng.cluster_valid(index_id, new_cid) is True
    sz0 = eng.cluster_live_size(index_id, 0)
    sz1 = eng.cluster_live_size(index_id, new_cid)
    assert sz0 + sz1 == vecs.shape[0]
    assert sz0 > 0 and sz1 > 0
    print(f"[split_merge] Split cluster 0 into clusters 0 and {new_cid} with sizes {sz0} and {sz1}")

    # Verify no IDs were lost: search with nprobe=2 (search both clusters).
    _assert_search_finds_self(eng, index_id, vecs[:20], ids[:20], nprobe=2)
    print("[split_merge] Verified self-retrieval after split for 20 queries")

    # Merge back: merge the smaller into the larger.
    if sz0 >= sz1:
        into, from_ = 0, new_cid
    else:
        into, from_ = new_cid, 0
    print(f"[split_merge] Merging cluster {from_} (size={min(sz0, sz1)}) into {into} (size={max(sz0, sz1)})")
    eng.merge_clusters(index_id, into, from_)
    assert eng.cluster_valid(index_id, from_) is False
    assert eng.cluster_valid(index_id, into) is True
    assert eng.cluster_live_size(index_id, into) == vecs.shape[0]
    print(f"[split_merge] Merge complete; cluster {into} now has {vecs.shape[0]} vectors")

    # Verify again: now nprobe=1 is sufficient (single valid cluster should win),
    # but use 2 to ensure robustness even if centroids changed.
    _assert_search_finds_self(eng, index_id, vecs[:20], ids[:20], nprobe=2)
    print("[split_merge] Verified self-retrieval after merge for 20 queries")
    print("Split/merge test passed.")
    print("nlist:", eng.nlist_of(index_id),
          "live_size:", eng.cluster_live_size(index_id, into))
    eng.stop()
    print("[split_merge] AsyncEngine stopped; test completed successfully")


if __name__ == "__main__":
    main()

