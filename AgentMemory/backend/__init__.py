# AgentMemory/backend/__init__.py
from .base import MemoryBackend
# from .placeholder import PlaceholderBackend
# from .quake import QuakeBackend
# from .m3 import M3Backend
# from .amem import AMemBackend

__all__ = ["PlaceholderBackend", "MemoryBackend", "QuakeBackend", "M3Backend", "AMemBackend", "FaissBackend", "MilvusBackend"]

def __getattr__(name: str):
    """Lazy import backends to avoid circular imports."""
    if name == "PlaceholderBackend":
        from .placeholder import PlaceholderBackend
        return PlaceholderBackend
    elif name == "QuakeBackend":
        from .quake import QuakeBackend
        return QuakeBackend
    elif name == "M3Backend":
        from .m3 import M3Backend
        return M3Backend
    elif name == "AMemBackend":
        from .amem import AMemBackend
        return AMemBackend
    elif name == "FaissBackend":
        from .faiss_backend import FaissBackend
        return FaissBackend
    elif name == "MilvusBackend":
        from .milvus import MilvusBackend
        return MilvusBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")