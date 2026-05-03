from __future__ import annotations

from typing import Any

from src.config import config
from src.retriever.query_analyzer import QueryAnalyzer
from src.vectorstore.embedder import Embedder
from src.vectorstore.store import VectorStore


class Retriever:
    """
    Handles retrieval of relevant documents from the vector store based on a natural language query.
    """

    def __init__(self) -> None:
        self.embedder = Embedder.from_config()
        self.store = VectorStore()
        self.query_analyzer = QueryAnalyzer()

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        override_filters: dict | None = None,
        debug: bool = False,
    ) -> list[dict]:
        """
            Retrieve relevant documents from the vector store based on the input query.
            Args:
                query: The natural language query string.
                top_k: Optional override for the number of top results to return. If None, uses default from config.
                override_filters: Optional dict of filters to apply instead of query analysis. Example: {"doc_type": "transaction", "year": 2023}
                debug: If True, prints debug information about the retrieval process.
            Returns:
                A list of retrieved items, where each item is a dict containing at least 'text' and 'metadata' keys.
        """
        effective_top_k = top_k if top_k is not None else config.top_k

        embedding = self.embedder.embed_query(query)

        if override_filters is not None:
            filters = override_filters
            if debug:
                print(f"Selected filters (override): {filters}")
        else:
            filters = self.query_analyzer.analyze(query)
            if debug:
                explanation = self.query_analyzer.explain(query)
                print(explanation)

        results = self.store.query(
            embedding=embedding,
            top_k=effective_top_k,
            filters=filters if filters else None,
        )

        if debug:
            print("Retrieved results:")
            for idx, item in enumerate(results, start=1):
                print(f"[{idx}] score={item.get('score')}")
                print(f"text: {item.get('text')}")
                print(f"metadata: {item.get('metadata')} \n")

        return results

    def format_context(self, results: list[dict]) -> str:
        chunks: list[str] = []

        for item in results:
            metadata: dict[str, Any] = item.get("metadata") or {}
            doc_type = metadata.get("doc_type", "unknown")
            text = item.get("text", "")

            meta_fields = self._format_metadata_fields(doc_type, metadata)
            header = f"[Source: {doc_type}"
            if meta_fields:
                header += f" | {meta_fields}"
            header += "]"

            chunks.append(f"{header}\n{text}\n---")

        return "\n".join(chunks)

    @staticmethod
    def _format_metadata_fields(doc_type: str, metadata: dict[str, Any]) -> str:
        doc_type_fields = {
            "transaction": ["year", "month", "category", "region", "segment"],
            "yearly_summary": ["year"],
            "monthly_summary": ["year", "month"],
            "quarterly_summary": ["year", "quarter"],
            "category_summary": ["category"],
            "subcategory_summary": ["subcategory"],
            "regional_yearly_summary": ["year", "region"],
            "quarterly_region_summary": ["quarter", "region"],
            "regional_summary": ["region"],
            "region_category_summary": ["region", "category"],
            "yearly_category_summary": ["year", "category"],
            "seasonality_summary": ["season"],
            "seasonality_pattern_overall": [],
            "comparative_category": [],
            "comparative_regional": [],
            "comparative_segment": [],
            "comparative_yearly": [],
            "comparative_discount_impact": [],
        }

        fields = doc_type_fields.get(doc_type, [])
        parts = []
        for field in fields:
            value = metadata.get(field)
            if value is None or value == "":
                continue
            parts.append(f"{field}={value}")
        return ", ".join(parts)
