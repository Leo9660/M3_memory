# AgentMemory/profiler.py
"""
M3 profile logger — Python side.

Enabled by setting env var M3_PROFILE=1.
Log file path: M3_PROFILE_LOG (default: m3_profile.log in CWD).

Appends structured key=value lines (same format as the C++ M3Profiler)
so the full encode→index pipeline can be analysed in one file.

Thread-safe: a module-level lock serialises all writes.
"""
from __future__ import annotations

import os
import threading
from datetime import datetime
from typing import Optional

_lock = threading.Lock()
_fp: Optional[object] = None
_enabled: bool = False


def _init() -> None:
    global _fp, _enabled
    if not os.environ.get("M3_PROFILE"):
        return
    path = os.environ.get("M3_PROFILE_LOG", "m3_profile.log")
    try:
        _fp = open(path, "a", buffering=1)   # line-buffered
        _fp.write("# M3 Profile Log (Python)  path={}\n".format(path))
        _fp.write("# Fields: [timestamp]  EVENT  key=value ...\n#\n")
        _fp.flush()
        _enabled = True
    except OSError as exc:
        import sys
        print(f"[M3Profiler] WARN: could not open profile log '{path}': {exc}",
              file=sys.stderr)


_init()


def is_enabled() -> bool:
    return _enabled


def _ts() -> str:
    return datetime.now().strftime("[%Y-%m-%d %H:%M:%S.%f")[:-3] + "]"


def _write(line: str) -> None:
    if not _enabled or _fp is None:
        return
    with _lock:
        _fp.write(f"{_ts()}  {line}\n")


# ---------------------------------------------------------------------------
# Encode batch
# ---------------------------------------------------------------------------

def log_encode_batch(
    *,
    batch: int,
    tokenize_ms: float,
    forward_ms: float,
    postprocess_ms: float,
    total_ms: float,
    mode: str = "items",   # "items" | "queries"
) -> None:
    """Log one encode call (tokenize → model forward → mean-pool/normalize)."""
    _write(
        f"PROFILE_ENCODE"
        f"  mode={mode:<8}"
        f"  batch={batch:<6}"
        f"  tokenize_ms={tokenize_ms:<10.3f}"
        f"  forward_ms={forward_ms:<10.3f}"
        f"  postprocess_ms={postprocess_ms:<10.3f}"
        f"  total_ms={total_ms:<10.3f}"
    )


# ---------------------------------------------------------------------------
# Backend insert / search
# ---------------------------------------------------------------------------

def log_backend_insert(
    *,
    batch: int,
    coerce_ms: float,
    index_ms: float,
    total_ms: float,
) -> None:
    """Log one backend INSERT execute() call."""
    _write(
        f"PROFILE_BACKEND_INSERT"
        f"  batch={batch:<6}"
        f"  coerce_ms={coerce_ms:<8.3f}"
        f"  index_ms={index_ms:<8.3f}"
        f"  total_ms={total_ms:<8.3f}"
    )


def log_backend_search(
    *,
    batch: int,
    flush_ms: float,
    search_ms: float,
    result_build_ms: float,
    total_ms: float,
) -> None:
    """Log one backend SEARCH execute() call."""
    _write(
        f"PROFILE_BACKEND_SEARCH"
        f"  batch={batch:<6}"
        f"  flush_ms={flush_ms:<8.3f}"
        f"  search_ms={search_ms:<8.3f}"
        f"  result_build_ms={result_build_ms:<8.3f}"
        f"  total_ms={total_ms:<8.3f}"
    )
