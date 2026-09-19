"""Persistent Chroma storage and semantic retrieval for document chunks.

Chunks are embedded with ``all-MiniLM-L6-v2`` and stored in the local
``data/chroma_db`` directory. Queries use the same embedding model, ensuring
that documents and questions share a compatible vector representation.
"""

import sys
import pysqlite3

sys.modules["sqlite3"] = pysqlite3

import chromadb
from sentence_transformers import SentenceTransformer


class VectorStore:
    """Manage the project's persistent Chroma collection and embedding model.

    The collection is created on demand as ``enterprise_documents``. Creating a
    new instance reconnects to the same on-disk database rather than creating a
    separate in-memory index.
    """

    def __init__(self):
        """Initialize the sentence-transformer model and Chroma collection."""
        self.embedding_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        self.client = chromadb.PersistentClient(
            path="data/chroma_db"
        )

        self.collection = self.client.get_or_create_collection(
            name="enterprise_documents"
        )

    def add_documents(self, chunks):
        """Embed and upsert normalized chunks into the persistent collection.

        Args:
            chunks: Dictionaries containing chunk ``text`` and ``metadata``.
                Metadata must include ``source``; the ingestion chunker supplies
                this alongside chunk identifiers and other provenance fields.

        The deterministic IDs make repeated ingestion replace matching indexed
        records instead of blindly appending duplicates.
        """
        texts = [chunk["text"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]

        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=True
        ).tolist()

        ids = [
            f"{metadata['source']}_{index}"
            for index, metadata in enumerate(metadatas)
        ]

        self.collection.upsert(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings,
        )

        print(f"Stored {len(texts)} chunks in Chroma")

    def search(self, query, top_k=3):
        """Return the ``top_k`` chunks most semantically similar to ``query``.

        Args:
            query: Natural-language question or search phrase.
            top_k: Maximum number of nearest documents to retrieve.

        Returns:
            Chroma's query response, including nested ``documents``,
            ``metadatas``, and ``distances`` lists.
        """
        query_embedding = self.embedding_model.encode(
            query
        ).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        return results
