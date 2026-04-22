from __future__ import annotations

from vector_index import index_suppliers


def main() -> None:
    result = index_suppliers(force=True)
    print("ChromaDB supplier index build complete")
    print(f"Status: {result.get('status')}")
    print(f"Collection: {result.get('collection')}")
    print(f"Source: {result.get('source')}")
    print(f"Indexed records: {result.get('indexed', result.get('records', 0))}")


if __name__ == "__main__":
    main()
