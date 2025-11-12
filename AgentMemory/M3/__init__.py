# Lightweight package init for M3.
from .wrapper.m3_async import M3AsyncEngine, Metric
from .faiss_loader import rebuild_from_faiss

__all__ = ["M3AsyncEngine", "Metric", "rebuild_from_faiss"]
