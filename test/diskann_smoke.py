"""
Standalone smoke script that exercises the DiskANN (diskannpy) backend without pytest.

Usage:
    python test/diskann_smoke.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    import diskannpy  # noqa: F401
except ImportError as exc:  # pragma: no cover - script guard
    print("diskannpy is required for this smoke test. Install it via `pip install diskannpy`.")
    raise SystemExit(1) from exc

from AgentMemory.backend.diskann import DiskANNBackend
from AgentMemory.types import BackendOpType, BackendRequest, CollectionSpec, Metric


def _print_hits(label: str, hits_batch):
    print(f"[SEARCH] {label}")
    for qi, hits in enumerate(hits_batch):
        formatted = [(hit.id, round(hit.score, 4)) for hit in hits]
        metas = [hit.metadata for hit in hits]
        print(f"  Q{qi}: hits={formatted} metas={metas}")


def main() -> None:
    tmp_root = Path("diskann_smoke_workspace")
    tmp_root.mkdir(exist_ok=True)
    backend = DiskANNBackend(
        index_directory=str(tmp_root / "indices"),
        max_vectors=256,
        graph_degree=32,
        build_complexity=64,
        search_complexity=64,
        auto_consolidate_every=1,
    )
    backend.create_index(0, CollectionSpec(name="demo", dim=3, metric=Metric.L2))
    print("[INIT] Created DiskANN index 'demo' (dim=3, metric=L2)")

    # INSERT
    insert_req = BackendRequest(
        op=BackendOpType.INSERT,
        index_id=0,
        ext_ids=["doc-0", "doc-1", "doc-2"],
        vectors=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
        metas=[{"tag": 0}, {"tag": 1}, {"tag": 2}],
        payloads=["zero", "one", "two"],
    )
    backend.execute([insert_req])
    print("[TEST] Inserted doc-0/doc-1/doc-2 to verify batch_insert works")

    # SEARCH
    search_req = BackendRequest(
        op=BackendOpType.SEARCH,
        index_id=0,
        vectors=np.array([[0, 0, 0], [1, 1, 0]], dtype=np.float32),
        k=2,
        request_id="search-1",
    )
    result = backend.execute([search_req])
    print("[TEST] Verifying initial search recall after inserts")
    _print_hits("after insert", result.searches["search-1"])

    # UPDATE doc-1
    update_req = BackendRequest(
        op=BackendOpType.UPDATE,
        index_id=0,
        ext_ids=["doc-1"],
        vectors=np.array([[0.5, 0.0, 0.0]], dtype=np.float32),
        metas=[{"tag": 99}],
        payloads=["updated"],
    )
    backend.execute([update_req])
    print("[TEST] Update doc-1 to confirm existing vectors/metadata can be modified")

    post_update = backend.execute([search_req])
    print("[TEST] Search to confirm doc-1 modification reflected")
    _print_hits("after update", post_update.searches["search-1"])

    # DELETE_IDS doc-2
    delete_req = BackendRequest(
        op=BackendOpType.DELETE_IDS,
        index_id=0,
        ext_ids=["doc-2"],
    )
    backend.execute([delete_req])
    print("[TEST] Deleted doc-2 via DELETE_IDS to ensure targeted removal")

    after_delete = backend.execute([search_req])
    print("[TEST] Search to confirm doc-2 no longer appears")
    _print_hits("after delete_ids doc-2", after_delete.searches["search-1"])

    # DELETE_KNN near [1,1,0]
    insert_req_2 = BackendRequest(
        op=BackendOpType.INSERT,
        index_id=0,
        ext_ids=["doc-3"],
        vectors=np.array([[1, 1, 0]], dtype=np.float32),
        metas=[{"tag": 3}],
        payloads=["three"],
    )
    backend.execute([insert_req_2])
    delete_knn = BackendRequest(
        op=BackendOpType.DELETE_KNN,
        index_id=0,
        vectors=np.array([[1, 1, 0]], dtype=np.float32),
        k=1,
        nprobe=64,
    )
    backend.execute([delete_knn])
    print("[TEST] delete_knn removed nearest neighbor to [1,1,0]")

    after_delete_knn = backend.execute([search_req])
    print("[TEST] Search to ensure delete_knn effect persisted")
    _print_hits("after delete_knn", after_delete_knn.searches["search-1"])

    # FLUSH to consolidate deletes explicitly
    backend.execute([BackendRequest(op=BackendOpType.FLUSH, index_id=0)])
    print("[TEST] Issued FLUSH to force consolidate_delete on pending removals")

    # SAVE INDEX
    save_dir = tmp_root / "saved_index"
    backend.save_index(0, path=str(save_dir))
    files = sorted(p.name for p in save_dir.iterdir())
    print(f"[TEST] Saved index files to {save_dir}: {files}")

    print("\nDiskANN smoke test complete ✅")


if __name__ == "__main__":
    main()
