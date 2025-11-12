AgentMemory
===========

AgentMemory glues a Hugging Face encoder to multiple vector index backends (`placeholder`, `quake`, `m3`). The interface layer encodes every `MemoryItem` in a single pass, serializes them into `BackendRequest`s, and replays that queue strictly in order for the selected backend. The project is built with `scikit-build-core`, so `pip install -e .` installs both the Python package and the local C++/pybind components.

Repository Tour
---------------

| Path | Description |
| --- | --- |
| `AgentMemory/` | Python interface layer, encoder definitions, backend shims. The `M3/` subpackage hosts the pybind + C++ sources. |
| `quake/` | Optional Quake engine sources (you can also `pip install quake-vector`). |
| `test/` | Integration scripts such as rebuilding from a Faiss checkpoint. |
| `build.sh` | Convenience wrapper for `python -m pip install -e .`. |
| `pyproject.toml` | `scikit-build-core` configuration for wheels and extensions. |

System Requirements
-------------------

1. Python ≥ 3.10 with `pip` and `venv`.
2. Modern C/C++ toolchain (`gcc`/`g++` 11+ or Clang), `cmake` ≥ 3.26, and `ninja` to build `AgentMemory.M3`.
3. CUDA 12.4 toolkit if you plan to use `torch==2.5.0+cu124` and the GPU backends.
4. Optional: `faiss` (rebuild from IVF/flat checkpoints) and `quake-vector` if you prefer using the published wheel instead of building `./quake`.

Python Environment
------------------

> GPU users typically need the PyTorch extra index (`--extra-index-url https://download.pytorch.org/whl/cu124`). For CPU-only installs, drop the `+cu124` suffix and pull the CPU wheels instead.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install --extra-index-url https://download.pytorch.org/whl/cu124 -r requirements.txt
./build.sh            # or python -m pip install -e .
```

`pip install -e .` triggers `scikit-build-core`, which:

* builds the `AgentMemory.M3._m3_async` pybind module (requires CMake + Ninja),
* installs the `AgentMemory` Python package (interface, encoder, and backend bindings).

Configuration & Build Notes
---------------------------

* **Backend selection**: `MemoryManagement(backend="placeholder"|"quake"|"m3")`. `placeholder` is dependency-free and good for smoke tests; `quake` requires an importable `quake` package; `m3` uses the local extension built above.
* **Encoder settings**: defaults to `intfloat/e5-large-v2`. Tune `model_name`, `precision`, `device`, `normalize`, `batch_size`, etc. Models download via Hugging Face conventions, so configure `HF_HOME` or proxies if needed.
* **ID / hashing**: `hash_mode` (`none`/`blake2b64`/`sha1_64`/`python`) and `auto_id_strategy` (`uuid`/`sequential`) are handled in the interface; backends only see normalized IDs plus vectors.
* **Faiss rebuild**: both `QuakeBackend.rebuild_index_from_faiss(...)` and `M3Backend.rebuild_index_from_faiss(...)` can restore an IVF index directly. See `python test/load_faiss_example.py --faiss-path ...` for a walkthrough.
* **Helpful env vars**: `CMAKE_BUILD_PARALLEL_LEVEL` controls build parallelism, `PYTORCH_CUDA_ALLOC_CONF` tunes CUDA memory behavior, and `TOKENIZERS_PARALLELISM=false` hides tokenizer warnings.

Usage Example
-------------

```python
from AgentMemory.interface import MemoryManagement
from AgentMemory.types import MemoryItem, Metric

mm = MemoryManagement(
    backend="placeholder",        # swap to "quake" or "m3" when ready
    model_name="intfloat/e5-large-v2",
    batch_size=16,
)

index_id = mm.create_index("demo", metric=Metric.COSINE)
mm.add_insert(index_id, [
    MemoryItem(id=None, data="LLMs remember tools", metadata={"tag": "doc1"}),
])
mm.add_search(index_id, [
    MemoryItem(id=None, data="What do LLMs remember?")
], k=3, request_id="search-1")

result = mm.run()
print(result.upserted, result.searches["search-1"][0][0].score)
```

Testing & Debugging
-------------------

* Smoke test:
  ```bash
  python - <<'PY'
  from AgentMemory.interface import MemoryManagement
  mm = MemoryManagement(backend="placeholder")
  mm.create_index("smoke")
  mm.run()
  PY
  ```
* Integration helpers: `python test/faiss_nq_pipeline.py --help` (requires extra deps such as `datasets`, `faiss`).
* Build logs land in `build/` (default `scikit-build-core` output). If compilation fails, inspect `build/*/CMakeError.log` or run `pip install -v -e .` for more detail.

Troubleshooting
---------------

1. **`fatal: detected dubious ownership`** – add the repo to Git’s safe list: `git config --global --add safe.directory /workspace/memory`.
2. **PyTorch/torchvision install fails** – ensure you used the CUDA extra index or switch to CPU wheels (`pip install torch==2.5.0 torchvision==0.20.0`).
3. **C++ build errors** – verify `cmake`, `ninja`, and `g++`/`clang++` are available; on GPU boxes also install the CUDA 12.4 headers.
4. **`quake` cannot be imported** – install `quake-vector` from PyPI or `cd quake && pip install -e .`.
