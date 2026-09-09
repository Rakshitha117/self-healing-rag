from app.retrieval.vector_store import VectorStore
from app.generation.llm import generate_answer


def main():

    question = input("Enter your question: ")

    vector_store = VectorStore()

    results = vector_store.search(
        query=question,
        top_k=3
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    context_parts = []

    for document, metadata in zip(documents, metadatas):

        context_parts.append(
            f"""
Source: {metadata.get('source')}
Page: {metadata.get('page')}

Content:
{document}
"""
        )

    context = "\n".join(context_parts)

    answer = generate_answer(
        question,
        context
    )

    print("\n" + "=" * 60)
    print("ANSWER")
    print("=" * 60)
    print(answer)


if __name__ == "__main__":
    main()