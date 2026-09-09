# Gen_AI_Training

## Project layout

- `data/raw/` — original website crawl files; do not edit these manually.
- `data/processed/` — generated datasets ready for the RAG pipeline.
- `scripts/prepare_rag_data.py` — removes cookie-banner-only page records.
- `scripts/` — data utility scripts.
- `examples/prompting/` — prompting practice examples.
- `apps/` — runnable application scripts.

## Prepare the RAG data

Run the cleaner from the project root:

```powershell
python scripts/prepare_rag_data.py
```

It removes cookie-banner records, normalizes excess whitespace while retaining
paragraph breaks, and writes `data/processed/vm_website_rag.json`. The raw
source file remains unchanged.
