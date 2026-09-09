"""Create a cleaned, RAG-ready ValueMomentum website dataset.

The raw crawl is never changed. Run this file to create the processed JSON:
    python scripts/prepare_rag_data.py
"""

import json
from pathlib import Path
import re


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "vm_website_copy.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "vm_website_rag.json"
COOKIE_BANNER_MARKER = 'By clicking "Accept" or continuing to use our website'


def is_cookie_banner_record(record: object) -> bool:
    """Return True when a crawled page was reduced to cookie-banner text."""
    return (
        isinstance(record, dict)
        and isinstance(record.get("text"), str)
        and COOKIE_BANNER_MARKER in record["text"]
    )


def normalize_page_text(text: str) -> str:
    """Normalize extracted text while retaining blank-line paragraph breaks.

    Website crawls commonly split one sentence across several lines. A single
    newline is therefore replaced with a space; one or more blank lines remain
    a paragraph break (``\n\n``). Repeated spaces are collapsed as well.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = re.split(r"\n\s*\n+", text)
    normalized_paragraphs = [
        re.sub(r"\s+", " ", paragraph).strip()
        for paragraph in paragraphs
    ]
    return "\n\n".join(paragraph for paragraph in normalized_paragraphs if paragraph)


def clean_website_records(records: list[object]) -> list[object]:
    """Remove cookie-banner pages and normalize text in retained page records."""
    cleaned_records = []

    for record in records:
        if is_cookie_banner_record(record):
            continue

        if isinstance(record, dict) and isinstance(record.get("text"), str):
            # Copy the record so this function never mutates the raw data loaded
            # from disk; only the generated RAG dataset receives normalized text.
            record = {**record, "text": normalize_page_text(record["text"])}

        cleaned_records.append(record)

    return cleaned_records


def prepare_rag_data(
    input_path: Path = INPUT_PATH,
    output_path: Path = OUTPUT_PATH,
) -> list[object]:
    """Read raw website JSON and write its cleaned copy for the RAG pipeline."""
    with input_path.open("r", encoding="utf-8") as file:
        records = json.load(file)

    if not isinstance(records, list):
        raise ValueError("Expected the source JSON to contain a list of page records.")

    cleaned_records = clean_website_records(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(cleaned_records, file, indent=2, ensure_ascii=False)

    print(f"Read records: {len(records)}")
    print(f"Removed cookie-banner records: {len(records) - len(cleaned_records)}")
    print(f"RAG-ready records: {len(cleaned_records)}")
    print("Normalized single line breaks and repeated whitespace in page text.")
    print(f"Output: {output_path}")
    return cleaned_records


if __name__ == "__main__":
    prepare_rag_data()
