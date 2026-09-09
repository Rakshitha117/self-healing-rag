from app.retrieval.vector_store import VectorStore


def search_documents(query, top_k=3):
    vector_store = VectorStore()

    results = vector_store.search(
        query=query,
        top_k=top_k
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for i, (document, metadata, distance) in enumerate(
        zip(documents, metadatas, distances)
    ):
        print("\n" + "=" * 60)
        print(f"Result {i + 1}")
        print(f"Distance: {distance}")
        print(f"Metadata: {metadata}")
        print(f"Text:\n{document}")


if __name__ == "__main__":
    query = input("Enter your question: ")
    search_documents(query)