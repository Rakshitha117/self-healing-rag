import sys
import pysqlite3

sys.modules["sqlite3"] = pysqlite3

import chromadb
from sentence_transformers import SentenceTransformer


class VectorStore:
    def __init__(self):
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
        query_embedding = self.embedding_model.encode(
            query
        ).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        return results