from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path when running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.vectorstore.embedder import Embedder
from src.vectorstore.store import VectorStore


def main() -> None:
    try:
        store = VectorStore()
        total_docs = store.count()
        print(f"Total documents in collection: {total_docs}")

        print("\nSample documents (collection.peek(3)):")
        peek_result = store.collection.peek(3)
        documents = peek_result.get("documents", [])
        metadatas = peek_result.get("metadatas", [])

        if not documents:
            print("- No documents available to preview.")
        else:
            for i, (text, metadata) in enumerate(zip(documents, metadatas), start=1):
                print(f"\n[{i}] text:")
                print(text)
                print(f"[{i}] metadata: {metadata}")

        print("\nRaw similarity query: 'technology sales performance' (top_k=3)")
        embedder = Embedder.from_config()
        query_embedding = embedder.embed_query("technology sales performance")
        results = store.query(embedding=query_embedding, top_k=3)

        if not results:
            print("- No results returned.")
        else:
            for i, item in enumerate(results, start=1):
                print(f"\nResult {i} | score={item['score']}")
                print(f"text: {item['text']}")
                print(f"metadata: {item['metadata']}")

    except Exception as exc:
        print(f"Verification failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
