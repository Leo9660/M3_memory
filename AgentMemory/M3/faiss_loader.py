from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from AgentMemory.M3 import _m3_async as m3

__all__ = ["rebuild_from_faiss"]


def _metric_from_faiss(faiss_module, metric_type: int) -> m3.Metric:
    if metric_type == faiss_module.METRIC_L2:
        return m3.Metric.L2
    if metric_type == faiss_module.METRIC_INNER_PRODUCT:
        return m3.Metric.IP
    cosine_attr = getattr(faiss_module, "METRIC_Cosine", None)
    if cosine_attr is not None and metric_type == cosine_attr:
        return m3.Metric.COSINE
    raise ValueError(f"Unsupported Faiss metric type: {metric_type}")


def rebuild_from_faiss(
    engine: m3.AsyncEngine,
    *,
    index_id: int,
    path: str | Path,
    normalized: bool = True,
) -> None:
    """
    Load an IVF index dumped by Faiss and rebuild the corresponding AsyncEngine index in-place.
    This keeps the existing AsyncEngine design unchanged (no new constructors), and relies on the
    new load_cluster() hook to populate per-cluster payloads.
    """

    try:
        import faiss  # type: ignore
        from faiss.contrib.inspect_tools import get_invlist  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("faiss is required to rebuild an index from a Faiss file") from exc

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Faiss index file not found: {path}")

    index = faiss.read_index(str(path))
    ivf = faiss.extract_index_ivf(index)
    if ivf is None:
        raise ValueError("Provided index does not contain an IVF component")
    ivf = faiss.downcast_index(ivf)

    metric = _metric_from_faiss(faiss, ivf.metric_type)
    dim = ivf.d
    nlist = ivf.nlist

    quantizer = faiss.downcast_index(ivf.quantizer)
    if hasattr(quantizer, "xb") and quantizer.ntotal == nlist:
        centroids = faiss.vector_to_array(quantizer.xb).astype(np.float32).reshape(nlist, dim)
    else:
        centroids = np.vstack([quantizer.reconstruct(i) for i in range(nlist)]).astype(np.float32)

    engine.create_ivf(int(index_id), int(dim), metric, bool(normalized), centroids)

    invlists = faiss.downcast_InvertedLists(ivf.invlists)
    for list_id in range(nlist):
        list_ids, list_codes = get_invlist(invlists, list_id)

        if list_ids.size == 0:
            continue
        if list_codes.dtype != np.uint8:
            raise ValueError("Only IndexIVFFlat (float codes) is supported for now")
        vectors = list_codes.view(np.float32).reshape(list_ids.shape[0], dim)
        engine.load_cluster(
            int(index_id),
            int(list_id),
            np.ascontiguousarray(list_ids, dtype=np.int64),
            np.ascontiguousarray(vectors, dtype=np.float32),
        )
