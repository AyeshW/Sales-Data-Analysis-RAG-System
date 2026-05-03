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

    print(f"Median text length: {median_len}")
    print(f"Average text length: {average_len}")
    print(f"Max text length: {max_len}")


if __name__ == "__main__":
    main()
