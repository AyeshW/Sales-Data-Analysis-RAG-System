import os
from dotenv import load_dotenv


class Config:
    def __init__(self) -> None:
        # Load environment variables from .env into process environment.
        load_dotenv()

        self.ollama_base_url: str = self._get_required("OLLAMA_BASE_URL")
        self.model_name: str = self._get_required("MODEL_NAME")

        self.embedding_model: str = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
        self.chroma_persist_dir: str = os.getenv("CHROMA_PERSIST_DIR", "chroma_store")
        self.collection_name: str = os.getenv("COLLECTION_NAME", "sales_rag")
        self.top_k: int = self._get_int("TOP_K", default=10)

    @staticmethod
    def _get_required(key: str) -> str:
        value = os.getenv(key)
        if value is None or value.strip() == "":
            raise ValueError(
                f"Missing required environment variable: {key}. "
                "Set it in your .env file before starting the application."
            )
        return value

    @staticmethod
    def _get_int(key: str, default: int) -> int:
        raw = os.getenv(key)
        if raw is None or raw.strip() == "":
            return default
        try:
            return int(raw)
        except ValueError as exc:
            raise ValueError(
                f"Invalid integer value for environment variable {key}: {raw!r}"
            ) from exc


config = Config()
