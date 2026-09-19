"""Cross-encoder reranking for improving initial vector-search results."""

from sentence_transformers import CrossEncoder


class Reranker:
    """Score query-document pairs with a cross-encoder and order them by fit.

    This is intended as a second retrieval stage after vector search: semantic
    search finds a broad candidate set cheaply, then the cross-encoder evaluates
    each question and candidate together for more precise relevance.
    """

    def __init__(self):
        """Load the MS MARCO MiniLM cross-encoder used for relevance scoring."""
        self.model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

    def rerank(self, query, documents, metadatas, top_k=3):
        """Return the highest-scoring document, metadata, score tuples.

        Args:
            query: User question used to assess relevance.
            documents: Candidate texts, usually from vector search.
            metadatas: Provenance dictionaries aligned with ``documents``.
            top_k: Number of highest-scoring candidates to retain.

        Returns:
            A descending list of ``(document, metadata, score)`` tuples.
        """
        pairs = [
            [query, document]
            for document in documents
        ]

        scores = self.model.predict(pairs)

        ranked_results = sorted(
            zip(documents, metadatas, scores),
            key=lambda x: x[2],
            reverse=True
        )

        return ranked_results[:top_k]
