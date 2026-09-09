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


from langchain_text_splitters import RecursiveCharacterTextSplitter


CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


def chunk_documents(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            ". ",
            "? ",
            "! ",
            "; ",
            ", ",
            " ",
            "",
        ],
    )

    all_chunks = []

    for document in documents:

        text = document["text"].strip()

        if not text:
            continue

        chunks = splitter.split_text(text)

        for chunk_index, chunk in enumerate(chunks):

            metadata = document["metadata"].copy()

            metadata["chunk_id"] = chunk_index
            metadata["chunk_size"] = len(chunk)

            all_chunks.append(
                {
                    "text": chunk,
                    "metadata": metadata,
                }
            )

    return all_chunks