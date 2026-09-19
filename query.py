from app.retrieval.vector_store import VectorStore
from app.retrieval.reranker import Reranker
from app.generation.llm import generate_answer


def main():
    question = input("Enter your question: ")

    vector_store = VectorStore()
    reranker = Reranker()

    # Step 1: Retrieve candidate documents
    results = vector_store.search(
        query=question,
        top_k=5
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    # Step 2: Rerank candidates
    ranked_results = reranker.rerank(
        query=question,
        documents=documents,
        metadatas=metadatas,
        top_k=3
    )

    # Step 3: Build context for LLM
    context_parts = []

    for document, metadata, score in ranked_results:
        context_parts.append(
            f"""
Source: {metadata.get('source')}
Page: {metadata.get('page', 'N/A')}

Content:
{document}
"""
        )

    context = "\n".join(context_parts)

    # Step 4: Generate answer
    answer = generate_answer(
        question=question,
        context=context
    )

    print("\n" + "=" * 60)
    print("ANSWER")
    print("=" * 60)
    print(answer)


if __name__ == "__main__":
    main()