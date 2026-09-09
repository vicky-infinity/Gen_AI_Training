"""Inspect cookie-banner records in the raw website crawl without changing it."""

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "vm_website_copy.json"
COOKIE_BANNER_MARKER = 'By clicking "Accept" or continuing to use our website'


def find_cookie_banner_records(records: list[object]) -> list[dict]:
    """Return page records whose extracted text includes the cookie banner."""
    return [
        record
        for record in records
        if isinstance(record, dict)
        and isinstance(record.get("text"), str)
        and COOKIE_BANNER_MARKER in record["text"]
    ]


if __name__ == "__main__":
    with INPUT_PATH.open("r", encoding="utf-8") as file:
        pages = json.load(file)

    matches = find_cookie_banner_records(pages)
    print(f"Cookie-banner records: {len(matches)}")
    for page in matches:
        print(page.get("url", "<missing URL>"))
