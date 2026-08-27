# test_amem_full.py
import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
a_mem_path = Path(__file__).parent.parent.parent / "A-mem"
if str(a_mem_path) not in sys.path:
    sys.path.insert(0, str(a_mem_path))

from AgentMemory.interface import MemoryManagement
from AgentMemory.types import MemoryItem, Metric

print("=" * 60)
print("A-MEM INTEGRATION TEST")
print("=" * 60)

# Step 1: Initialize MemoryManagement with amem backend
print("\n[Step 1] Initializing MemoryManagement with 'amem' backend...")
try:
    mm = MemoryManagement(
        backend="amem",
        model_name="intfloat/e5-large-v2",  # This will be used by interface layer
        # A-mem will use its own encoder internally
    )
    print("✓ MemoryManagement initialized")
except Exception as e:
    print(f"✗ Initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 2: Create an index
print("\n[Step 2] Creating index...")
try:
    index_id = mm.create_index(handle="test_index", metric=Metric.COSINE)
    print(f"✓ Index created with id: {index_id}")
except Exception as e:
    print(f"✗ Index creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 3: Test INSERT
print("\n[Step 3] Testing INSERT operation...")
try:
    mm.add_insert("test_index", [
        MemoryItem(
            id="doc1",
            data="Machine learning is a subset of artificial intelligence that focuses on algorithms.",
            metadata={"category": "AI", "source": "test"}
        ),
        MemoryItem(
            id="doc2",
            data="Neural networks are computing systems inspired by biological neural networks.",
            metadata={"category": "AI", "source": "test"}
        ),
        MemoryItem(
            id="doc3",
            data="Python is a high-level programming language known for its simplicity.",
            metadata={"category": "programming", "source": "test"}
        ),
    ])
    result = mm.run()
    print(f"✓ INSERT completed")
    print(f"  - Upserted: {result.upserted}")
    print(f"  - Updated: {result.updated}")
    print(f"  - Deleted: {result.deleted}")
except Exception as e:
    print(f"✗ INSERT failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 4: Test SEARCH
print("\n[Step 4] Testing SEARCH operation...")
try:
    mm.add_search(
        "test_index",
        [MemoryItem(id=None, data="What is machine learning?", metadata={"type": "query"})],
        k=2,
        request_id="search-1"
    )
    result = mm.run()
    print(f"✓ SEARCH completed")
    
    if "search-1" in result.searches:
        search_result = result.searches["search-1"]
        if len(search_result) > 0:
            hits = search_result[0]  # First query's results
            if len(hits) > 0:
                print(f"  - Found {len(hits)} results:")
                for i, hit in enumerate(hits, 1):
                    print(f"    {i}. id={hit.id}, score={hit.score:.4f}")
                    if hit.metadata:
                        print(f"       content: {hit.metadata.get('content', 'N/A')[:60]}...")
            else:
                print("  ⚠ Query returned 0 results")
        else:
            print("  ⚠ Search returned empty result list (no queries processed)")
    else:
        print("  ⚠ Search request_id 'search-1' not found in results")
except Exception as e:
    print(f"✗ SEARCH failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 5: Test UPDATE
print("\n[Step 5] Testing UPDATE operation...")
try:
    mm.add_update("test_index", [
        MemoryItem(
            id="doc1",
            data="Machine learning is a powerful subset of AI that enables systems to learn from data.",
            metadata={"category": "AI", "source": "test", "updated": True}
        ),
    ])
    result = mm.run()
    print(f"✓ UPDATE completed")
    print(f"  - Updated: {result.updated}")
except Exception as e:
    print(f"✗ UPDATE failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 6: Test SEARCH after UPDATE
print("\n[Step 6] Testing SEARCH after UPDATE...")
try:
    mm.add_search(
        "test_index",
        [MemoryItem(id=None, data="machine learning", metadata={"type": "query"})],
        k=1,
        request_id="search-2"
    )
    result = mm.run()
    if "search-2" in result.searches:
        search_result = result.searches["search-2"]
        if len(search_result) > 0:
            hits = search_result[0]
            if len(hits) > 0:
                print(f"✓ SEARCH after UPDATE found result:")
                print(f"  - id={hits[0].id}, score={hits[0].score:.4f}")
                if hits[0].metadata:
                    content = hits[0].metadata.get('content', '')
                    print(f"  - content preview: {content[:80]}...")
            else:
                print("  ⚠ Query returned 0 results")
        else:
            print("  ⚠ Search returned empty result list")
    else:
        print("  ⚠ Search request_id 'search-2' not found in results")
except Exception as e:
    print(f"✗ SEARCH after UPDATE failed: {e}")
    import traceback
    traceback.print_exc()

# Step 7: Test DELETE_IDS
print("\n[Step 7] Testing DELETE_IDS operation...")
try:
    mm.add_delete_ids("test_index", ["doc3"])
    result = mm.run()
    print(f"✓ DELETE_IDS completed")
    print(f"  - Deleted: {result.deleted}")
except Exception as e:
    print(f"✗ DELETE_IDS failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 8: Test SEARCH after DELETE
print("\n[Step 8] Testing SEARCH after DELETE...")
try:
    mm.add_search(
        "test_index",
        [MemoryItem(id=None, data="Python programming", metadata={"type": "query"})],
        k=3,
        request_id="search-3"
    )
    result = mm.run()
    if "search-3" in result.searches:
        search_result = result.searches["search-3"]
        if len(search_result) > 0:
            hits = search_result[0]
            if len(hits) > 0:
                print(f"✓ SEARCH after DELETE found {len(hits)} results")
                print(f"  - doc3 should be deleted (not in results)")
                for hit in hits:
                    print(f"    - id={hit.id}")
            else:
                print("  ⚠ Query returned 0 results")
        else:
            print("  ⚠ Search returned empty result list")
    else:
        print("  ⚠ Search request_id 'search-3' not found in results")
except Exception as e:
    print(f"✗ SEARCH after DELETE failed: {e}")
    import traceback
    traceback.print_exc()

# Step 9: Test DELETE_KNN
print("\n[Step 9] Testing DELETE_KNN operation...")
try:
    # First add a few more documents
    mm.add_insert("test_index", [
        MemoryItem(id="doc4", data="Deep learning uses neural networks with multiple layers."),
        MemoryItem(id="doc5", data="Natural language processing helps computers understand human language."),
    ])
    mm.run()
    
    # Now delete k nearest to a query
    mm.add_delete_knn(
        "test_index",
        [MemoryItem(id=None, data="neural networks", metadata={"type": "query"})],
        k=1
    )
    result = mm.run()
    print(f"✓ DELETE_KNN completed")
    print(f"  - Deleted: {result.deleted}")
except Exception as e:
    print(f"✗ DELETE_KNN failed: {e}")
    import traceback
    traceback.print_exc()

# Step 10: Verify final state
print("\n[Step 10] Final verification...")
try:
    mm.add_search(
        "test_index",
        [MemoryItem(id=None, data="AI and machine learning", metadata={"type": "query"})],
        k=5,
        request_id="search-final"
    )
    result = mm.run()
    if "search-final" in result.searches:
        search_result = result.searches["search-final"]
        if len(search_result) > 0:
            hits = search_result[0]
            if len(hits) > 0:
                print(f"✓ Final search found {len(hits)} results")
                print("  Remaining documents:")
                for hit in hits:
                    print(f"    - {hit.id}: score={hit.score:.4f}")
            else:
                print("  ⚠ Query returned 0 results")
        else:
            print("  ⚠ Search returned empty result list")
    else:
        print("  ⚠ Search request_id 'search-final' not found in results")
except Exception as e:
    print(f"✗ Final verification failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)