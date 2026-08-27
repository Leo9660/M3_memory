# test_amem_init.py
import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
a_mem_path = Path(__file__).parent.parent.parent / "A-mem"
if str(a_mem_path) not in sys.path:
    sys.path.insert(0, str(a_mem_path))

from AgentMemory.backend.amem import AMemBackend
from AgentMemory.types import CollectionSpec, Metric

print("Testing AMemBackend initialization...")

try:
    backend = AMemBackend()
    print("✓ Backend created successfully")
except Exception as e:
    print(f"✗ Backend creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test create_index
try:
    spec = CollectionSpec(name="test_index", dim=384, metric=Metric.COSINE)
    backend.create_index(index_id=0, spec=spec)
    print("✓ Index created successfully")
except Exception as e:
    print(f"✗ Index creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✓ Initialization test passed!")