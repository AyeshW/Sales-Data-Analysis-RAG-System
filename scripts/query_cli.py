from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import requests

# Ensure project root is on sys.path when running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import config
from src.pipeline.rag_chain import RAGChain
from src.vectorstore.store import VectorStore

SEPARATOR = "=" * 80
SUB_SEPARATOR = "-" * 80

COMMANDS = {
    "/exit or /quit": "Exit the program",
    "/debug": "Toggle debug mode on/off",
    "/model": "Show the active model name",
    "/count": "Show collection document count",
    "/help": "Show available commands",
}


def print_help() -> None:
    print("Available commands:")
    for command, description in COMMANDS.items():
        print(f"  {command:<15} {description}")


def print_banner(doc_count: int) -> None:
    print(SEPARATOR)
    print("  Sales Data Analysis RAG CLI")
    print("  - Ask questions about the Superstore")
    print(SUB_SEPARATOR)
    print(f"Active model: {config.model_name}")
    print(f"Collection documents: {doc_count}")
    print_help()
    print(SEPARATOR)


def confirm_ollama_or_exit() -> None:
    base_url = config.ollama_base_url.rstrip("/")
    url = f"{base_url}/api/tags"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
    except Exception as exc:
        print(
            "Error: Unable to reach Ollama at "
            f"{config.ollama_base_url!r}. "
            "Ensure Ollama is running and accessible."
        )
        print(f"Details: {exc}")
        sys.exit(1)


def run_query(rag_chain: RAGChain, question: str, debug_mode: bool) -> None:
    print("\nRetrieving and generating...\n")

    def _long_running_notice() -> None:
        print("I'm still working on it. Please wait...\n")

    timer = threading.Timer(20.0, _long_running_notice)
    timer.daemon = True
    timer.start()

    try:
        start_time = time.perf_counter()
        result = rag_chain.run(question, debug=debug_mode)
        elapsed = time.perf_counter() - start_time
        answer = result.get("answer", "")
        print("Answer:\n")
        print(answer)
        print(f"\nResponse time: {elapsed:.2f}s")
    except Exception as exc:
        print(f"Error: {exc}")
    finally:
        timer.cancel()
        print(SUB_SEPARATOR)


def main() -> None:
    confirm_ollama_or_exit()

    store = VectorStore()
    doc_count = store.count()
    print_banner(doc_count)

    rag_chain = RAGChain()
    debug_mode = False

    while True:
        try:
            user_input = input("\n> ").strip()
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        lowered = user_input.lower()
        if lowered in ("/exit", "/quit"):
            print("Goodbye!")
            break
        if lowered == "/debug":
            debug_mode = not debug_mode
            state = "ON" if debug_mode else "OFF"
            print(f"Debug mode: {state}")
            continue
        if lowered == "/model":
            print(f"Active model: {config.model_name}")
            continue
        if lowered == "/count":
            print(f"Collection documents: {doc_count}")
            continue
        if lowered == "/help":
            print_help()
            continue

        run_query(rag_chain, user_input, debug_mode)


if __name__ == "__main__":
    main()
