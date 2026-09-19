"""Generate grounded answers from retrieved evidence using a local Ollama LLM."""

import ollama


MODEL_NAME = "llama3.2"


def generate_answer(question, context):
    """Ask the configured model to answer a question using only supplied context.

    Args:
        question: The user's natural-language question.
        context: Retrieved document text and provenance formatted by the caller.

    Returns:
        The assistant message returned by Ollama.

    The prompt explicitly requires an insufficiency response when the retrieved
    evidence does not answer the question, limiting unsupported answers.
    """

    prompt = f"""
You are an enterprise knowledge assistant.

Answer the user's question using ONLY the provided context.

If the answer is not present in the context, say:
"I don't have enough information in the provided documents."

Context:
{context}

Question:
{question}

Answer:
"""

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]
