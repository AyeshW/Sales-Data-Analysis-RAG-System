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

PREDEFINED_QUESTIONS = [
    "What is the sales trend over the 4-year period?",
    "Which months show the highest sales? Is there seasonality?",
    "How does the West region compare to the East in terms of profit?",
    "Which product category generates the most revenue?",
    "What sub-categories have the highest profit margins?",
    "Which region has the best sales performance?",
    "Which segment drives the most orders and highest profit margin?",
]

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


def print_question_menu() -> None:
    print("  ════════════════════════════════════════════════════════════════")
    print("  Example Questions:")
    for index, question in enumerate(PREDEFINED_QUESTIONS, start=1):
        print(f"    [{index}]  {question}")
    print("  ════════════════════════════════════════════════════════════════")
    print("  Enter a number to pick a question, or type your own question.")
    print("  Commands: /exit  /debug  /model  /count  /help")


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
        print(f"### Answer: \n{answer}\n")
        print(f"Response time: {elapsed:.2f}s")
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
    show_menu = True

    while True:
        try:
            if show_menu:
                print_question_menu()
                user_input = input("  > ").strip()
            else:
                user_input = input("\n  > ").strip()
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break

        if not user_input:
            show_menu = False
            continue

        lowered = user_input.lower()
        if lowered in ("/exit", "/quit"):
            print("Goodbye!")
            break
        if lowered == "/debug":
            debug_mode = not debug_mode
            state = "ON" if debug_mode else "OFF"
            print(f"Debug mode: {state}")
            show_menu = False
            continue
        if lowered == "/model":
            print(f"Active model: {config.model_name}")
            show_menu = False
            continue
        if lowered == "/count":
            print(f"Collection documents: {doc_count}")
            show_menu = False
            continue
        if lowered == "/help":
            print_help()
            show_menu = False
            continue

        if user_input.isdigit():
            selection = int(user_input)
            if 1 <= selection <= len(PREDEFINED_QUESTIONS):
                question = PREDEFINED_QUESTIONS[selection - 1]
                print(f"Question: {question}")
                run_query(rag_chain, question, debug_mode)
                show_menu = True
            else:
                print(f"Please enter a number between 1 and {len(PREDEFINED_QUESTIONS)}.")
                show_menu = False
            continue

        run_query(rag_chain, user_input, debug_mode)
        show_menu = True


if __name__ == "__main__":
    main()
