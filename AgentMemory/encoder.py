# AgentMemory/encoder.py
from __future__ import annotations
import time
from abc import ABC, abstractmethod
from typing import Any, List, Optional
import json
import numpy as np
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

from .types import MemoryItem
from .profiler import is_enabled as _prof_enabled, log_encode_batch as _prof_encode

DEFAULT_MODEL = "intfloat/e5-large-v2"

# ---------- model loader ----------
def load_model(
    model_path: str,
    *,
    use_fp16: bool = False,
    device: Optional[str] = None,
    trust_remote_code: bool = True,
):
    """
    Load HF model + tokenizer with trust_remote_code and optional FP16.
    Moves model to CUDA by default when available.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("[encoder] loading model:", model_path)

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    model.eval()
    model.to(device)

    if use_fp16 and (device.startswith("cuda")):
        model = model.half()  # FP16 weights on CUDA

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        trust_remote_code=trust_remote_code,
    )
    return model, tokenizer, device, config


# ---------- base encoder ----------
class MemoryEncoder(ABC):
    """
    Base encoder for text/vector transformation using Hugging Face Transformers.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        precision: str = "fp32",        # "fp32" | "fp16" | "bf16"
        dim: Optional[int] = None,      # optional output dim override
        device: Optional[str] = None,   # "cuda", "cpu", "cuda:0"
        normalize: bool = True,
        batch_size: int = 32,
        max_length: int = 512,
        trust_remote_code: bool = True,
    ) -> None:
        self.model_name = model_name
        self.normalize = normalize
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)

        # Precision flags
        prec = precision.lower()
        use_fp16 = prec == "fp16"
        use_bf16 = prec == "bf16"

        # Load model/tokenizer using your pattern
        model, tok, device_str, config = load_model(
            model_name,
            use_fp16=use_fp16,
            device=device,
            trust_remote_code=trust_remote_code,
        )

        # Optional BF16 cast (if supported)
        if use_bf16 and device_str.startswith("cuda"):
            try:
                model = model.to(dtype=torch.bfloat16)
            except Exception:
                # Fallback to FP16 if BF16 is not supported
                model = model.half()

        self.model = model
        self.tokenizer = tok
        self.device = torch.device(device_str)
        self.config = config

        # Determine native hidden size and set output dim
        native_dim = int(getattr(self.config, "hidden_size", 1024))
        self.native_dim = native_dim
        self.dim = int(dim) if dim is not None else native_dim

        # Disable grad
        for p in self.model.parameters():
            p.requires_grad_(False)

    @abstractmethod
    def encode_items(self, items: List[MemoryItem]) -> np.ndarray:
        """Encode MemoryItems into (N, D) float32 embeddings."""
        ...

    @abstractmethod
    def encode_queries(self, items: List[MemoryItem]) -> np.ndarray:
        """Encode query MemoryItems into (Q, D) float32 embeddings."""
        ...

    # ---------- utilities ----------
    def _coerce_to_str(self, data: Any) -> str:
        """Convert any payload to a stable UTF-8 string for tokenization."""
        if isinstance(data, np.ndarray):
            return json.dumps(data.tolist(), ensure_ascii=False)
        if isinstance(data, (list, tuple)):
            return json.dumps(list(data), ensure_ascii=False)
        if isinstance(data, dict):
            return json.dumps(data, sort_keys=True, ensure_ascii=False)
        return str(data)

    @torch.no_grad()
    def _encode_texts(self, texts: List[str], _mode: str = "items") -> np.ndarray:
        """
        Tokenize, forward through the model, mean-pool with attention mask,
        resize (truncate/pad) to self.dim, and L2-normalize if requested.
        """
        profiling = _prof_enabled()
        t_total = time.perf_counter() if profiling else 0.0
        t_tok_total = t_fwd_total = t_post_total = 0.0

        all_vecs: List[np.ndarray] = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i : i + self.batch_size]

            # --- tokenize ---
            t0 = time.perf_counter() if profiling else 0.0
            toks = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            toks = {k: v.to(self.device) for k, v in toks.items()}
            if profiling: t_tok_total += time.perf_counter() - t0

            # --- model forward ---
            t0 = time.perf_counter() if profiling else 0.0
            out = self.model(**toks)  # last_hidden_state: [B, T, H]
            last = out.last_hidden_state
            if profiling:
                # Synchronize before stopping the timer so GPU work is actually
                # complete — otherwise lazy CUDA kernels flush inside .cpu() and
                # inflate postprocess_ms instead.
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                t_fwd_total += time.perf_counter() - t0

            # --- mean pooling + to numpy ---
            t0 = time.perf_counter() if profiling else 0.0
            mask = toks["attention_mask"].unsqueeze(-1).type_as(last)  # [B, T, 1]
            summed = (last * mask).sum(dim=1)                           # [B, H]
            counts = mask.sum(dim=1).clamp(min=1e-6)                    # [B, 1]
            emb = summed / counts                                       # [B, H]
            emb = emb.detach().cpu().to(torch.float32).numpy()          # -> np.float32
            all_vecs.append(emb)
            if profiling: t_post_total += time.perf_counter() - t0

        # --- dim resize + L2 normalize (postprocess) ---
        t0 = time.perf_counter() if profiling else 0.0
        X = (
            np.vstack(all_vecs)
            if all_vecs
            else np.zeros((0, self.native_dim), dtype=np.float32)
        )

        # Adjust to requested output dim
        if X.shape[1] != self.dim:
            if X.shape[1] > self.dim:
                X = X[:, : self.dim]
            else:
                pad = np.zeros((X.shape[0], self.dim - X.shape[1]), dtype=np.float32)
                X = np.concatenate([X, pad], axis=1)

        # L2 normalize
        if self.normalize and X.size > 0:
            n = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
            X = X / n
        if profiling: t_post_total += time.perf_counter() - t0

        if profiling:
            total_ms = (time.perf_counter() - t_total) * 1000
            _prof_encode(
                batch=len(texts),
                tokenize_ms=t_tok_total * 1000,
                forward_ms=t_fwd_total * 1000,
                postprocess_ms=t_post_total * 1000,
                total_ms=total_ms,
                mode=_mode,
            )

        return X.astype("float32")


class TransformerEncoder(MemoryEncoder):
    """
    HF Transformers–based encoder.
    Converts MemoryItem.data to string and encodes via mean-pooled hidden states.
    """

    def encode_items(self, items: List[MemoryItem]) -> np.ndarray:
        texts = [self._coerce_to_str(it.data) for it in items]
        return self._encode_texts(texts, _mode="items")

    def encode_queries(self, items: List[MemoryItem]) -> np.ndarray:
        texts = [self._coerce_to_str(it.data) for it in items]
        return self._encode_texts(texts, _mode="queries")
