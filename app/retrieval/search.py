from app.retrieval.vector_store import VectorStore
from app.retrieval.reranker import Reranker


def search_documents(query, retrieval_k=5, rerank_k=3):

    vector_store = VectorStore()
    reranker = Reranker()

    results = vector_store.search(
        query=query,
        top_k=retrieval_k
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    ranked_results = reranker.rerank(
        query=query,
        documents=documents,
        metadatas=metadatas,
        top_k=rerank_k
    )

    print("\n" + "=" * 60)
    print("RERANKED RESULTS")
    print("=" * 60)

    for index, (document, metadata, score) in enumerate(
        ranked_results
    ):

        print(f"\nResult {index + 1}")
        print(f"Score: {score:.4f}")
        print(f"Metadata: {metadata}")
        print(f"Text:\n{document}")


if __name__ == "__main__":
    query = input("Enter your question: ")
    search_documents(query)