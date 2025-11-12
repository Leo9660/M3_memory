"""
Example QA retrieval loop:
  1. Rebuild the M3 backend from a Faiss IVF checkpoint.
  2. Load a JSON document store to map ids -> documents (order-preserving if the JSON is a list).
  3. Stream the `nq` subset of the FlashRAG dataset (`RUC-NLPIR/FlashRAG_datasets`) and run searches.
  4. Optionally insert freshly processed docs (preserves legacy ingestion flow).
  5. Print similarity, ids, and the matched document text for each query.

Run with:
    python test/faiss_nq_pipeline.py --limit 50 --top-k 5 \
        --faiss-path /path/to/index.faiss \
        --doc-json /path/to/doc_store.json

Requirements: datasets, faiss (if --faiss-path is used), and the E5 encoder weights.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple
from datasets import load_dataset  # type: ignore

# Ensure repo-root imports work when script is executed from test/.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from AgentMemory.interface import MemoryManagement
from AgentMemory.types import MemoryItem, Metric


def load_tevatron_msmarco_records(
    *,
    split: str,
    limit: Optional[int] = None,
) -> List[Tuple[str, str, Optional[Dict[str, Any]]]]:
    """
    Load passages from Tevatron/msmarco-passage-corpus (HF), returning list of (doc_id, text, metadata).
    """
    dataset = load_dataset("Tevatron/msmarco-passage-corpus", split=split, revision="0.0.0")
    print(
        f"Loaded Tevatron/msmarco-passage-corpus (revision 0.0.0) split='{split}' "
        f"with {len(dataset)} rows and columns {dataset.column_names}"
    )
    records: List[Tuple[str, str, Optional[Dict[str, Any]]]] = []
    seen: set[str] = set()
    for idx, sample in enumerate(dataset):
        text = sample.get("text")
        if not text:
            continue
        doc_id = sample.get("text_id") or sample.get("id") or idx
        doc_id = str(doc_id)
        if doc_id in seen:
            continue
        seen.add(doc_id)
        title = sample.get("title")
        metadata: Optional[Dict[str, Any]] = {"title": title} if title else None
        records.append((doc_id, text, metadata))
        if limit is not None and len(records) >= limit:
            break
    return records


def load_natural_questions(limit: int) -> List[Tuple[str, str]]:
    print("Loading FlashRAG NQ dataset...")
    dataset = load_dataset("RUC-NLPIR/FlashRAG_datasets", "nq", split="train")
    print("Loaded FlashRAG NQ dataset with", len(dataset), "rows")
    pairs: List[Tuple[str, str]] = []
    for i, row in enumerate(dataset):
        question = row.get("question")
        answers = row.get("golden_answers")
        if not question or not answers:
            continue
        if isinstance(answers, list):
            answer_text = answers[0] if answers else ""
        else:
            answer_text = str(answers)
        if not answer_text:
            continue
        pairs.append((question, answer_text))
        if len(pairs) >= limit:
            break
    print(f"Prepared {len(pairs)} question-answer pairs.")
    return pairs


def _summarize_doc(doc: Any, max_len: int = 160) -> str:
    if doc is None:
        return "<doc unavailable>"
    if isinstance(doc, str):
        text = doc
    else:
        try:
            text = json.dumps(doc, ensure_ascii=False)
        except TypeError:
            text = str(doc)
    text = " ".join(text.split())
    if len(text) > max_len:
        return f"{text[: max_len - 3]}..."
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="FlashRAG-NQ retrieval demo using MemoryManagement + Faiss rebuild.")
    parser.add_argument("--faiss-path", type=str, required=True, help="Faiss IVF index checkpoint to rebuild.")
    parser.add_argument("--doc-json", type=str, required=True, help="JSON file describing the document store.")
    parser.add_argument("--index-handle", type=str, default="nq-demo")
    parser.add_argument("--limit", type=int, default=10, help="Number of QA pairs to process.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--nprobe", type=int, default=32, help="Search nprobe to use for each query.")
    parser.add_argument("--doc-hf-split", type=str, default="train", help="HF split to use when doc-json names Tevatron/msmarco.")
    parser.add_argument(
        "--doc-hf-limit",
        type=int,
        default=None,
        help="Optional max docs when loading from a Hugging Face dataset (default loads all).",
    )
    parser.add_argument("--backend", type=str, default="m3", choices=["m3", "quake", "placeholder"], help="MemoryManagement backend to use.",)
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="After each search, insert the QA pair into the index (legacy behavior).",
    )
    args = parser.parse_args()

    print("Loading NQ dataset into memory...")
    question_answer_pairs = load_natural_questions(limit=args.limit)

    print("initializing MemoryManagement...")

    mm = MemoryManagement(backend=args.backend, model_name="intfloat/e5-large-v2")
    mm.create_index(handle=args.index_handle, metric=Metric.L2)

    faiss_path = Path(args.faiss_path).expanduser().resolve()
    if not faiss_path.is_file():
        raise FileNotFoundError(f"Faiss checkpoint not found: {faiss_path}")
    mm.rebuild_index_from_faiss(args.index_handle, path=str(faiss_path))
    print(f"Rebuilt index '{args.index_handle}' from {faiss_path}")

    doc_spec = args.doc_json.strip()
    if doc_spec.lower() == "tevatron/msmarco-passage-corpus":
        print(f"Loading doc store from Hugging Face dataset {doc_spec} ...")
        records = load_tevatron_msmarco_records(
            split=args.doc_hf_split,
            limit=args.doc_hf_limit,
        )
        # preview = records[:10]
        # for idx, (doc_id, text, meta) in enumerate(preview, start=1):
        #     print(
        #         f"[Tevatron/MSMARCO] doc#{idx} id={doc_id} title={_summarize_doc(meta.get('title') if meta else None)} "
        #         f"text={_summarize_doc(text)}"
        #     )
        mm.load_doc_store_from_records(records)
        print(f"Loaded {mm.doc_store_size()} documents from Hugging Face dataset {doc_spec}")
    else:
        doc_json_path = Path(doc_spec).expanduser().resolve()
        if not doc_json_path.is_file():
            raise FileNotFoundError(f"Doc store JSON not found: {doc_json_path}")
        mm.load_doc_store(str(doc_json_path))
        print(f"Loaded {mm.doc_store_size()} documents from {doc_json_path}")

    search_time_total = 0.0
    search_count = 0
    insert_time_total = 0.0
    insert_count = 0

    for i, (question, answer) in enumerate(question_answer_pairs, start=1):

        doc_payload = f"Question: {question} Answer: {answer}"

        print(f"\nProcessing question {i}: {question}")
        print(f"Ground-truth answer: {answer}")

        request_id = f"qa-{i}"
        mm.add_search(
            args.index_handle,
            [MemoryItem(id=None, data=question, metadata={"type": "query"})],
            k=args.top_k,
            request_id=request_id,
            nprobe=args.nprobe,
        )

        start = time.perf_counter()
        result = mm.run()
        elapsed = time.perf_counter() - start
        search_time_total += elapsed
        search_count += 1
        hits = result.searches.get(request_id, [])
        top_hits = hits[0] if hits else []
        if top_hits:
            print("Top hits:")
            for rank, hit in enumerate(top_hits, start=1):
                doc_text = _summarize_doc(mm.get_doc(hit.id))
                print(f"  #{rank}: id={hit.id} score={hit.score:.4f} doc={doc_text}")
        else:
            print("Top hits: []")

        if args.ingest:
            mm.add_insert(
                args.index_handle,
                [
                    MemoryItem(
                        id=None,
                        data=doc_payload,
                        metadata={"answer": answer},
                    )
                ],
            )
            start = time.perf_counter()
            mm.run()
            elapsed_write = time.perf_counter() - start
            insert_time_total += elapsed_write
            insert_count += 1

    if search_count:
        print(f"\nAverage search time: {search_time_total / search_count:.4f}s over {search_count} queries")
    if insert_count:
        print(f"Average insert time: {insert_time_total / insert_count:.4f}s over {insert_count} inserts")


if __name__ == "__main__":
    main()
