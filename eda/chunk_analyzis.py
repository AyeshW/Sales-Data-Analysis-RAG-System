import json
from pathlib import Path

import numpy as np


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    chunks_path = base_dir / "data" / "datastore" / "chunks.json"

    with chunks_path.open("r", encoding="utf-8") as file:
        docs = json.load(file)

    if not docs:
        print("No docs found.")
        return

    lengths = [len(doc.get("text", "")) for doc in docs]

    if not lengths:
        print("No text fields found.")
        return

    median_len = float(np.median(lengths))
    average_len = float(np.mean(lengths))
    max_len = max(lengths)
    p90_len = float(np.percentile(lengths, 90))
    p95_len = float(np.percentile(lengths, 95))
    p98_len = float(np.percentile(lengths, 98))

    print(f"Median text length: {median_len}")
    print(f"Average text length: {average_len}")
    print(f"Max text length: {max_len}")
    print(f"90th percentile text length: {p90_len}")
    print(f"95th percentile text length: {p95_len}")
    print(f"98th percentile text length: {p98_len}")

    top_docs = sorted(
        enumerate(lengths),
        key=lambda item: item[1],
        reverse=True
    )[:10]

    print("Top 10 largest docs (by character count):")
    for rank, (idx, size) in enumerate(top_docs, start=1):
        print(f"  {rank}. doc_index={idx}, chars={size}")


if __name__ == "__main__":
    main()
