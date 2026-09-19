"""Ingest the documents in ``data/raw`` into the persistent vector store.

This script is the indexing entry point for the project. It loads every
supported source document, preserves source metadata, divides its content into
retrieval-sized chunks, creates embeddings, and upserts those chunks into the
Chroma collection used by the query workflow.
"""

from pathlib import Path

from app.ingestion.loader import load_document
from app.ingestion.chunker import chunk_documents
from app.retrieval.vector_store import VectorStore


DATA_FOLDER = Path("data/raw")


def main():
    """Load, chunk, preview, and index all documents in :data:`DATA_FOLDER`.

    A failure in one source file is reported but does not prevent the remaining
    files from being indexed. Re-running the script is safe because the vector
    store writes chunks with deterministic identifiers and uses Chroma upserts.
    """
    all_documents = []

    for file_path in DATA_FOLDER.iterdir():
        if file_path.is_file():
            try:
                documents = load_document(file_path)
                all_documents.extend(documents)

                print(f"Loaded: {file_path.name}")

            except Exception as error:
                print(f"Failed: {file_path.name}")
                print(f"Error: {error}")

    chunks = chunk_documents(all_documents)

    print(f"\nTotal documents: {len(all_documents)}")
    print(f"Total chunks: {len(chunks)}")

    for index, chunk in enumerate(chunks[:3]):
        print(f"\nChunk {index + 1}")
        print(chunk["metadata"])
        print(chunk["text"][:300])
    vector_store = VectorStore()
    vector_store.add_documents(chunks)


if __name__ == "__main__":
    main()
