import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from AgentMemory.types import Metric, MemoryItem
from AgentMemory.interface import MemoryManagement


def main():
    # m = MemoryManagement(backend=PlaceholderBackend())
    m = MemoryManagement(backend="m3")  # uses PlaceholderBackend by default

    # 创建 index "demo"
    m.create_index(handle="demo", metric=Metric.COSINE)

    # 插入两条数据
    m.add_insert("demo", [
        MemoryItem(id="t1", data="hello world", metadata={"role": "u"}),
        MemoryItem(id="t2", data="agent memory interface", metadata={"role": "sys"}),
    ])

    # 入队一次搜索（request_id 用于从结果中取回）
    rid1 = "s1"
    m.add_search(
        "demo",
        [MemoryItem(id=None, data="memory interface for agents")],
        k=3,
        request_id=rid1,
    )

    out = m.run()
    print("\n=== RUN #1 ===")
    print("mutations:", {"upserted": out.upserted, "updated": out.updated, "deleted": out.deleted})
    print("search keys:", list(out.searches.keys()))
    for i, hits in enumerate(out.searches[rid1]):
        print(f"Q{i} top-3:", [(h.id, round(h.score, 4)) for h in hits])

    # 更新、按ID删除、再搜索
    m.add_update("demo", [MemoryItem(id="t1", data={"updated": True, "content": "hello updated"})])

    # 注意：新接口为 add_delete_ids；若使用哈希键，需传入外层 id（interface 会按配置哈希）
    m.add_delete_ids("demo", ["v1"])  # 若不存在会被安全跳过

    rid2 = "s2"
    m.add_search("demo", [MemoryItem(id=None, data="hello updated content")], k=2, request_id=rid2)

    out2 = m.run()
    print("\n=== RUN #2 ===")
    print("mutations:", {"upserted": out2.upserted, "updated": out2.updated, "deleted": out2.deleted})
    for i, hits in enumerate(out2.searches[rid2]):
        print(f"Q{i} top-2:", [(h.id, round(h.score, 4)) for h in hits])
        # PlaceholderBackend 会把原始 payload 放到 metadata["_data"]
        print("hits metadata:", [h.metadata for h in hits])

if __name__ == "__main__":
    main()
