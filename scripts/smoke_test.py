from __future__ import annotations

import sys
import time
from pathlib import Path

# Ensure project root is on sys.path when running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.rag_chain import RAGChain


def main() -> None:
    questions = [
        "What is the overall sales trend from 2014 to 2017?",
        "Which product category generates the most revenue?",
        "How does the West region compare to the East in terms of profit?",
    ]

    rag = RAGChain()

    for question in questions:
        print("\n" + "=" * 80)
        print(f"Question: {question}")
        start_time = time.time()
        result = rag.run(question, debug=True)
        elapsed = time.time() - start_time
        print(f"Answer: {result.get('answer', '')}")
        print(f"Time taken: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
