from __future__ import annotations

from langchain_ollama import OllamaEmbeddings

from src.config import config


class Embedder:
    def __init__(self, embedding_model: str, ollama_base_url: str) -> None:
        self.embedding_model = embedding_model
        self.ollama_base_url = ollama_base_url

        self._embeddings = OllamaEmbeddings(
            model=self.embedding_model,
            base_url=self.ollama_base_url,
            num_thread=4,
        )

        # Connectivity check at initialization time.
        try:
            self._embeddings.embed_query("connection test")
        except Exception as exc:
            raise RuntimeError(
                "Unable to connect to Ollama for embeddings. "
                f"base_url={self.ollama_base_url!r}, model={self.embedding_model!r}. "
                "Ensure Ollama is running and the model is available."
            ) from exc

    @classmethod
    def from_config(cls) -> Embedder:
        return cls(
            embedding_model=config.embedding_model,
            ollama_base_url=config.ollama_base_url,
        )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return self._embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embeddings.embed_query(text)
