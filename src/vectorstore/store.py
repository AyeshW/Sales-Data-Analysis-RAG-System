from __future__ import annotations

from typing import Any

import chromadb
from tqdm import tqdm

from src.config import config


class VectorStore:
    def __init__(self) -> None:
        self.client = chromadb.PersistentClient(path=config.chroma_persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=config.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list[dict], embeddings: list[list[float]]) -> None:
        if self.collection.count() > 0:
            print(
                "Warning: collection already contains documents. "
                "Skipping ingestion to avoid duplicates."
            )
            return

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Length mismatch: got {len(chunks)} chunks and {len(embeddings)} embeddings."
            )

        batch_size = 500
        total = len(chunks)

        for start in tqdm(
            range(0, total, batch_size),
            desc="Ingesting chunks",
            unit="batch",
        ):
            end = min(start + batch_size, total)
            batch_chunks = chunks[start:end]
            batch_embeddings = embeddings[start:end]

            ids = [f"chunk_{i}" for i in range(start, end)]
            documents = [chunk["text"] for chunk in batch_chunks]
            metadatas = [
                {k: v for k, v in chunk["metadata"].items() if v is not None}
                for chunk in batch_chunks
            ]

            self.collection.add(
                ids=ids,
                documents=documents,
                embeddings=batch_embeddings,
                metadatas=metadatas,
            )

    def query(
        self, embedding: list[float], top_k: int, filters: dict | None = None
    ) -> list[dict]:
        query_kwargs: dict[str, Any] = {
            "query_embeddings": [embedding],
            "n_results": top_k,
        }

        if filters:
            query_kwargs["where"] = filters

        result = self.collection.query(**query_kwargs)

        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        return [
            {"text": doc, "metadata": meta, "score": dist}
            for doc, meta, dist in zip(documents, metadatas, distances)
        ]

    def count(self) -> int:
        return self.collection.count()

    def reset(self) -> None:
        print(
            "Warning: resetting vector store will delete the existing collection "
            "and all stored embeddings."
        )
        self.client.delete_collection(name=config.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=config.collection_name,
            metadata={"hnsw:space": "cosine", "hue_space": "cosine"},
        )
