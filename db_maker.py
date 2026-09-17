import shutil
import sys
import time
from pathlib import Path

from docx import Document as DocxDocument
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

DOCX_PATH = "valuemomentum_sample.docx"
PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "valuemomentum"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_docx_text(path: str) -> str:
    if not Path(path).exists():
        raise FileNotFoundError(f"DOCX file not found: '{path}'")
    doc = DocxDocument(path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    text = "\n".join(paragraphs)
    if not text.strip():
        raise ValueError(f"No readable text found in '{path}'")
    return text


def reset_persist_dir(path: str) -> None:
    persist_path = Path(path)
    if persist_path.exists():
        print(f"Existing DB found at '{path}', deleting for a fresh rebuild...")
        shutil.rmtree(persist_path)

# this is the main loop
def main():
    start = time.time()

    try:
        text = load_docx_text(DOCX_PATH)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)

    print(f"Loaded '{DOCX_PATH}' -> {len(text)} characters")
    print(f"Split into {len(chunks)} chunks\n")

    for i, chunk in enumerate(chunks, start=1):
        print(f"--- Chunk {i} ({len(chunk)} chars) ---")
        print(chunk)
        print()

    documents = [
        Document(page_content=chunk, metadata={"source": DOCX_PATH, "chunk": i})
        for i, chunk in enumerate(chunks, start=1)
    ]

    try:
        reset_persist_dir(PERSIST_DIR)
    except OSError as e:
        print(f"ERROR: could not clear existing DB folder '{PERSIST_DIR}': {e}")
        sys.exit(1)

    try:
        print(f"Loading embedding model '{EMBEDDING_MODEL}' (first run downloads the model, this can take a few minutes)...")
        t0 = time.time()
        from langchain_huggingface import HuggingFaceEmbeddings

        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        print(f"Embedding model ready in {time.time() - t0:.1f}s")
    except Exception as e:
        print(f"ERROR: failed to load embedding model '{EMBEDDING_MODEL}': {e}")
        sys.exit(1)

    try:
        print(f"Embedding {len(documents)} chunks and storing them in Chroma at '{PERSIST_DIR}'...")
        t0 = time.time()
        from langchain_chroma import Chroma

        vectordb = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            persist_directory=PERSIST_DIR,
        )
        count = vectordb._collection.count()
        print(f"Stored {count} vectors in {time.time() - t0:.1f}s")
    except Exception as e:
        print(f"ERROR: failed to build the Chroma vector DB: {e}")
        sys.exit(1)

    print(f"\nDB is created. Collection '{COLLECTION_NAME}' has {count} vectors in '{PERSIST_DIR}'.")
    print(f"Total time: {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
