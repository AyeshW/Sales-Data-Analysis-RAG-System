from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


ALLOWED_DOC_TYPES = {
    "transaction",
    "monthly_summary",
    "yearly_summary",
    "quarterly_summary",
    "category_summary",
    "regional_summary",
    "subcategory_summary",
    "seasonality_summary",
    "seasonality_pattern_overall",
    "comparative_category",
    "comparative_regional",
    "comparative_segment",
    "comparative_yearly",
    "comparative_discount_impact",
    "yearly_category_summary",
    "regional_yearly_summary",
    "region_category_summary",
    "quarterly_region_summary",
}


def load_chunks(filepath: str) -> list[dict]:
    """Load chunk records from a JSON file.

    Args:
        filepath: Path to a JSON file containing a top-level list of chunk dicts.

    Returns:
        Parsed list of dictionaries.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If JSON is invalid or top-level content is not a list.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Chunks file not found: {filepath}")

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in chunks file '{filepath}': {exc}") from exc

    if not isinstance(data, list):
        raise ValueError(
            f"Invalid chunk file format in '{filepath}': expected a JSON list at top level, "
            f"got {type(data).__name__}."
        )

    return data


def validate_chunks(chunks: list[dict]) -> None:
    """Validate chunk schema and print a doc_type summary.

    Validation rules:
      - each chunk has non-empty string `text`
      - each chunk has dict `metadata`
      - each `metadata` has `doc_type`
      - `doc_type` is one of ALLOWED_DOC_TYPES

    Raises:
        ValueError: If any validation errors are found.
    """
    errors: list[str] = []
    doc_type_counts: Counter[str] = Counter()

    for idx, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            errors.append(
                f"chunk[{idx}]: expected dict, got {type(chunk).__name__}."
            )
            continue

        text = chunk.get("text")
        if not isinstance(text, str) or not text.strip():
            errors.append(
                f"chunk[{idx}]: 'text' must be a non-empty string."
            )

        metadata = chunk.get("metadata")
        if not isinstance(metadata, dict):
            errors.append(
                f"chunk[{idx}]: 'metadata' must be a dict."
            )
            continue

        if "doc_type" not in metadata:
            errors.append(
                f"chunk[{idx}]: metadata is missing required key 'doc_type'."
            )
            continue

        doc_type = metadata.get("doc_type")
        if not isinstance(doc_type, str):
            errors.append(
                f"chunk[{idx}]: metadata['doc_type'] must be a string."
            )
            continue

        if doc_type not in ALLOWED_DOC_TYPES:
            allowed = ", ".join(sorted(ALLOWED_DOC_TYPES))
            errors.append(
                f"chunk[{idx}]: invalid doc_type '{doc_type}'. Allowed values: {allowed}."
            )
            continue

        doc_type_counts[doc_type] += 1

    if errors:
        details = "\n".join(f"- {error}" for error in errors)
        raise ValueError(f"Chunk validation failed with {len(errors)} error(s):\n{details}")

    print(f"Chunk validation successful. Total chunks: {len(chunks)}")
    print("doc_type breakdown:")
    for doc_type in sorted(doc_type_counts):
        print(f"- {doc_type}: {doc_type_counts[doc_type]}")
