from __future__ import annotations

from langchain_ollama import ChatOllama

from src.config import config
from src.pipeline.prompt_templates import ANALYTICAL_PROMPT, FALLBACK_PROMPT
from src.retriever.retriever import Retriever


class RAGChain:
    def __init__(self) -> None:
        self.retriever = Retriever()
        self.llm = ChatOllama(
            model=config.model_name,
            base_url=config.ollama_base_url,
            temperature=0.1,
        )
        self.analytical_prompt = ANALYTICAL_PROMPT
        self.fallback_prompt = FALLBACK_PROMPT

    def run(self, question: str, top_k: int | None = None, debug: bool = False) -> dict:
        results = self.retriever.retrieve(question, top_k=top_k, debug=debug)

        if not results:
            prompt = self.fallback_prompt.format(question=question)
            response = self.llm.invoke(prompt)
            answer = getattr(response, "content", str(response))
            return {
                "question": question,
                "answer": answer,
            }

        context = self.retriever.format_context(results)
        prompt = self.analytical_prompt.format(context=context, question=question)
        response = self.llm.invoke(prompt)
        answer = getattr(response, "content", str(response))

        return {
            "question": question,
            "answer": answer,
        }
