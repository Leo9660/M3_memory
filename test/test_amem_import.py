# test_amem_import.py (create in M3_memory/test/)
import sys
from pathlib import Path

# Add A-mem to path
a_mem_path = Path(__file__).parent.parent.parent / "A-mem"
if str(a_mem_path) not in sys.path:
    sys.path.insert(0, str(a_mem_path))

try:
    from memory_layer import AgenticMemorySystem
    print("✓ A-mem import successful")
    print(f"  AgenticMemorySystem: {AgenticMemorySystem}")
except Exception as e:
    print(f"✗ A-mem import failed: {e}")
    sys.exit(1)

# # Test M3 backend import
# try:
#     from AgentMemory.backend.amem import AMemBackend
#     print("✓ AMemBackend import successful")
# except Exception as e:
#     print(f"✗ AMemBackend import failed: {e}")
#     sys.exit(1)

print("\n✓ All imports successful!")