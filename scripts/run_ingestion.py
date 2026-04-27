from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from tqdm import tqdm

# Ensure project root is on sys.path when running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingest.loader import load_chunks, validate_chunks
from src.vectorstore.embedder import Embedder
from src.vectorstore.store import VectorStore


def main():
    parser = argparse.ArgumentParser(description="Run full ingestion pipeline")
    parser.add_argument(
        "--chunks-file",
        default="./data/chunks.json",
        help="Path to chunks JSON file (default: ./data/chunks.json)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset the collection before ingestion",
    )
    args = parser.parse_args()

    try:
        chunks = load_chunks(args.chunks_file)
        validate_chunks(chunks)

        store = VectorStore()
        if args.reset:
            store.reset()

        # Early-exit if already populated and not resetting
        current_count = store.count()
        if current_count > 0 and not args.reset:
            print(f"Current collection count: {current_count}")
            print("Collection already populated. Use --reset to re-ingest.")
            return

        embedder = Embedder.from_config()
        print("Ollama connectivity confirmed.")

        # Extract texts and embed all
        with tqdm(total=2, desc="Embedding pipeline", unit="step") as progress:
            texts = [chunk["text"] for chunk in chunks]
            progress.update(1)

            embeddings = embedder.embed_texts(texts)
            progress.update(1)

        store.add_chunks(chunks=chunks, embeddings=embeddings)

        print(f"Ingestion complete. {store.count()} documents stored.")

    except Exception as exc:
        print(f"Ingestion failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
