# from langchain_text_splitters import RecursiveCharacterTextSplitter


# def chunk_documents(documents):
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=800,
#         chunk_overlap=150,
#         separators=[
#             "\n\n",
#             "\n",
#             ". ",
#             " ",
#             "",
#         ],
#     )

#     all_chunks = []

#     for document in documents:
#         chunks = splitter.split_text(document["text"])

#         for chunk_index, chunk in enumerate(chunks):
#             metadata = document["metadata"].copy()
#             metadata["chunk_id"] = chunk_index

#             all_chunks.append(
#                 {
#                     "text": chunk,
#                     "metadata": metadata,
#                 }
#             )

#     return all_chunks


# Original Document
#        ↓
# Remove unnecessary whitespace
#        ↓
# RecursiveCharacterTextSplitter
#        ↓
# Try paragraph boundary
#        ↓
# Try line boundary
#        ↓
# Try sentence boundary
#        ↓
# Try smaller boundaries if necessary
#        ↓
# Create ~500-character chunks
#        ↓
# Keep 100-character overlap
#        ↓
# Attach metadata
#        ↓
# Return chunks

from langchain_text_splitters import RecursiveCharacterTextSplitter


# Maximum number of characters allowed in a single chunk.
# A chunk should be small enough for efficient embedding and retrieval,
# but large enough to preserve the meaning/context of the original text.
CHUNK_SIZE = 500

# Number of characters from the previous chunk that should be repeated
# in the next chunk.
#
# Overlap helps prevent important information from being lost when a
# sentence or piece of context falls across two chunk boundaries.
#
# Example:
# Chunk 1: "...employees can claim travel expenses within 15 days..."
# Chunk 2: "...within 15 days after the expense is incurred..."
#
# The overlapping text gives the retrieval system additional context.
CHUNK_OVERLAP = 100


def chunk_documents(documents):
    """
    Split each document into smaller, overlapping chunks.

    Why do we need chunking in RAG?

    Large documents cannot be directly embedded and retrieved efficiently.
    Instead, the document is divided into smaller pieces called "chunks".

    Each chunk is then:
        1. Converted into an embedding.
        2. Stored in the vector database.
        3. Retrieved later when a user's question is semantically similar
           to that chunk.

    Input:
        documents = [
            {
                "text": "Document content...",
                "metadata": {
                    "source": "policy.txt"
                }
            }
        ]

    Output:
        [
            {
                "text": "A smaller piece of the document...",
                "metadata": {
                    "source": "policy.txt",
                    "chunk_id": 0,
                    "chunk_size": 480
                }
            }
        ]
    """

    # RecursiveCharacterTextSplitter attempts to split the document
    # while preserving as much meaningful text structure as possible.
    #
    # Instead of blindly cutting the text every 500 characters, it tries
    # different separators in the order provided below.
    #
    # This is important because splitting at a paragraph or sentence
    # boundary generally preserves meaning better than splitting in the
    # middle of a sentence.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,

        # Separators are tried from the most meaningful boundary
        # to the least meaningful boundary.
        #
        # The splitter first tries to separate paragraphs.
        # If the resulting pieces are still too large, it progressively
        # moves to smaller boundaries such as sentences, words, and finally
        # individual characters.
        separators=[
            "\n\n",  # Paragraph boundary
            "\n",    # Line boundary
            ". ",    # End of a normal sentence
            "? ",    # End of a question
            "! ",    # End of an exclamation
            "; ",    # Semicolon boundary
            ", ",    # Comma boundary
            " ",     # Word boundary
            "",      # Last fallback: split at character level
        ],
    )

    # This list will contain the chunks generated from ALL documents.
    #
    # Each chunk keeps both:
    #   - the actual text
    #   - metadata describing where the text came from
    #
    # Keeping metadata is important in RAG because later we may want
    # to identify the source document, page number, chunk number, etc.
    all_chunks = []

    # Process every document individually.
    #
    # We do not combine all documents into one large text before chunking,
    # because each document needs to maintain its own identity and metadata.
    for document in documents:

        # Remove unnecessary whitespace from the beginning and end
        # of the document.
        #
        # Example:
        # "\n\n  Leave policy text  \n\n"
        # becomes:
        # "Leave policy text"
        text = document["text"].strip()

        # Skip documents that contain no meaningful text.
        #
        # This prevents empty strings from being passed to the splitter
        # and avoids creating useless chunks in the vector database.
        if not text:
            continue

        # Split the current document into smaller pieces.
        #
        # The splitter tries to keep each chunk close to CHUNK_SIZE while
        # respecting the separator hierarchy defined above.
        #
        # The result is a list of strings:
        #
        # [
        #     "Employees are entitled to...",
        #     "The employee must submit...",
        #     "Managers are responsible for..."
        # ]
        chunks = splitter.split_text(text)

        # Enumerate the generated chunks so that every chunk gets
        # a unique position/index within its source document.
        #
        # Example:
        # chunk_index = 0 -> first chunk
        # chunk_index = 1 -> second chunk
        # chunk_index = 2 -> third chunk
        for chunk_index, chunk in enumerate(chunks):

            # Copy the original document metadata.
            #
            # We use .copy() instead of modifying document["metadata"]
            # directly. This is important because each chunk needs its
            # own metadata dictionary.
            metadata = document["metadata"].copy()

            # Store the position of this chunk within the document.
            #
            # This is useful later for:
            #   - debugging retrieval
            #   - identifying which chunk was retrieved
            #   - tracing the answer back to the original document
            metadata["chunk_id"] = chunk_index

            # Store the actual character length of this chunk.
            #
            # This is useful for inspecting whether the chunking strategy
            # is producing chunks of the expected size.
            metadata["chunk_size"] = len(chunk)

            # Store the chunk text together with its metadata.
            #
            # The final structure is intentionally simple so that it can
            # later be passed to the embedding/vector-store pipeline.
            all_chunks.append(
                {
                    "text": chunk,
                    "metadata": metadata,
                }
            )

    # Return chunks from all processed documents.
    #
    # These chunks can now move to the next stage of the RAG pipeline:
    #
    #       Documents
    #           ↓
    #       Chunking              <-- THIS FUNCTION
    #           ↓
    #       Embeddings
    #           ↓
    #       Vector Database
    #           ↓
    #       Similarity Search
    #           ↓
    #       LLM
    return all_chunks