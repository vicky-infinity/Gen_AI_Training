"""Metadata-cleanup utility.

This legacy cleanup work is intentionally disabled so it cannot overwrite a
raw dataset by accident. Use ``prepare_rag_data.py`` for the active RAG-data
preparation flow.
"""

if __name__ == "__main__":
    print("Use: python scripts/prepare_rag_data.py")
