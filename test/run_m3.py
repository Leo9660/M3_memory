# test_m3_minimal.py
import numpy as np

from AgentMemory.M3 import _m3_async as m3


def main():
    eng = m3.AsyncEngine()

    # 启动后台线程（1个写线程，1个维护线程就够）
    eng.start(1, 1)

    # ===== 1) 创建一个 index =====
    index_id = 0
    dim = 1024
    metric = m3.Metric.COSINE
    normalized = True

    # 我们随便造 2 个 centroid，表示这个 IVF 里有 2 个 cluster
    nlist = 2
    centroids = np.random.rand(nlist, dim).astype(np.float32)

    # 这一行就是“必须先创建 index”
    eng.create_ivf(index_id, dim, metric, normalized, centroids)

    # ===== 2) 向 cluster 0 插入几条向量 =====
    # 注意：现在的接口是 “index_id + cluster_id”
    ids = np.array([101, 102, 103], dtype=np.int64)
    vecs = np.random.rand(3, dim).astype(np.float32)

    ok = eng.enqueue_insert(index_id, 0, ids, vecs)
    assert ok
    eng.flush()

    # ===== 3) 查一下 =====
    q = np.random.rand(1, dim).astype(np.float32)
    out_ids, out_scores = eng.search(index_id, q, 3)

    print(out_ids, out_scores)

    return

    print("query:", q)
    print("search ids:", out_ids)
    print("search scores:", out_scores)

    eng.stop()


if __name__ == "__main__":
    main()
