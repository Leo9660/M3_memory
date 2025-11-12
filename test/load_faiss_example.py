"""
Example script that rebuilds the native M3 AsyncEngine from a Faiss index file,
inserts an extra vector, and runs a search.  Usage:

    python test/load_faiss_example.py --faiss-path /path/to/index.faiss

Requirements:
    - numpy and faiss must be installed in the current environment.
    - The Faiss index must be IVF-based (IndexIVFFlat-style float codes).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np

# Make repository importable when running from the test directory.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from AgentMemory.M3 import rebuild_from_faiss  # noqa: E402  (import after sys.path tweak)
from AgentMemory.M3 import _m3_async as m3     # type: ignore # noqa: E402

def _load_faiss_metadata(index_path: Path):
    import faiss

    index = faiss.read_index(str(index_path))
    ivf = faiss.extract_index_ivf(index)
    if ivf is None:
        raise ValueError("Index does not contain an IVF component.")
    ivf = faiss.downcast_index(ivf)

    dim = int(ivf.d)
    metric_type = int(ivf.metric_type)
    nlist = int(ivf.nlist)
    ntotal = int(ivf.ntotal)

    quantizer = faiss.downcast_index(ivf.quantizer)
    if hasattr(quantizer, "xb") and quantizer.ntotal == nlist:
        centroids = faiss.vector_to_array(quantizer.xb).astype(np.float32).reshape(nlist, dim)
    else:
        centroids = np.vstack([quantizer.reconstruct(i) for i in range(nlist)]).astype(np.float32)

    meta = {"dim": dim, "metric_type": metric_type, "nlist": nlist, "ntotal": ntotal}
    return meta, centroids

def _assign_cluster(vec: np.ndarray, centroids: np.ndarray, metric_type: int) -> int:
    import faiss  # type: ignore

    if metric_type == faiss.METRIC_INNER_PRODUCT:
        scores = centroids @ vec
        return int(np.argmax(scores))
    # default to squared L2 / cosine
    diff = centroids - vec
    dists = np.sum(diff * diff, axis=1)
    return int(np.argmin(dists))


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild AsyncEngine from Faiss IVF index.")
    parser.add_argument("--faiss-path", required=True, help="Path to the Faiss IVF index file (.faiss)")
    parser.add_argument("--index-id", type=int, default=0, help="Logical index id to use inside the engine")
    parser.add_argument("--writer-threads", type=int, default=1)
    parser.add_argument("--maintenance-threads", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    np.random.seed(args.seed)
    index_path = Path(args.faiss_path).expanduser().resolve()
    if not index_path.is_file():
        raise FileNotFoundError(f"Faiss index file not found: {index_path}")

    ivf_meta, centroids = _load_faiss_metadata(index_path)
    dim = ivf_meta["dim"]
    metric_type = ivf_meta["metric_type"]
    print(f"Loaded Faiss IVF index (dim={dim}, nlist={ivf_meta['nlist']}, ntotal={ivf_meta['ntotal']}).")
    
    eng = m3.AsyncEngine()
    eng.start(args.writer_threads, args.maintenance_threads)

    rebuild_from_faiss(eng, index_id=args.index_id, path=index_path)
    print("Rebuilt AsyncEngine from Faiss index.")

    # Insert a synthetic vector that is close to centroid 0 so we can observe it in the search results.
    base_centroid = centroids[0]
    new_vec = (base_centroid + 0.01 * np.random.randn(dim)).astype(np.float32)
    new_id = int(10_000_000 + np.random.randint(1_000_000))
    target_cluster = _assign_cluster(new_vec, centroids, metric_type)
    print(f"Inserting doc id {new_id} into cluster {target_cluster}.")

    eng.enqueue_insert(
        int(args.index_id),
        int(target_cluster),
        np.array([new_id], dtype=np.int64),
        new_vec.reshape(1, -1),
    )
    eng.flush()

    # Run a quick search with two queries:
    queries = np.stack([new_vec, base_centroid], axis=0).astype(np.float32)
    ids, scores = eng.search(int(args.index_id), queries, int(args.top_k))

    print("\nSearch results:")
    for qi, (qid_list, score_list) in enumerate(zip(ids, scores)):
        pretty = ", ".join(f"(id={doc_id}, score={score:.4f})" for doc_id, score in zip(qid_list, score_list))
        print(f"  Query {qi}: {pretty}")

    eng.flush()
    eng.stop()


if __name__ == "__main__":
    main()
