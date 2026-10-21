from pathlib import Path
import re


DOCUMENT_PATH = Path("data/knowledge_base.txt")


def load_document():
    """Load the knowledge-base document."""

    if not DOCUMENT_PATH.exists():
        raise FileNotFoundError(
            f"Document not found: {DOCUMENT_PATH}"
        )

    return DOCUMENT_PATH.read_text(encoding="utf-8")


def split_into_chunks(text, chunk_size=50, overlap=10):
    """Split text into overlapping word-based chunks."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be non-negative and smaller than chunk_size."
        )

    words = text.split()
    chunks = []

    step = chunk_size - overlap

    for start in range(0, len(words), step):
        chunk_words = words[start:start + chunk_size]

        if not chunk_words:
            break

        chunks.append(" ".join(chunk_words))

        if start + chunk_size >= len(words):
            break

    return chunks


def tokenize(text):
    """Convert text into lowercase word tokens."""

    return re.findall(r"\b\w+\b", text.lower())


def retrieve_relevant_chunks(question, chunks, top_k=3):
    """Rank chunks using simple word-overlap scoring."""

    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    question_words = set(tokenize(question))

    if not question_words:
        return []

    scored_chunks = []

    for index, chunk in enumerate(chunks):
        chunk_words = set(tokenize(chunk))

        overlap = question_words.intersection(chunk_words)

        # Simple baseline relevance score.
        score = len(overlap) / len(question_words)

        scored_chunks.append({
            "chunk_id": index,
            "text": chunk,
            "score": score
        })

    scored_chunks.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    # Return only chunks with a positive relevance score.
    relevant = [
        item for item in scored_chunks
        if item["score"] > 0
    ]

    return relevant[:top_k]


def main():
    document = load_document()

    chunks = split_into_chunks(document)

    print(f"Document loaded successfully.")
    print(f"Total chunks: {len(chunks)}")

    question = input(
        "\nEnter a question about the document: "
    )

    results = retrieve_relevant_chunks(question, chunks)

    print("\n===== RETRIEVAL RESULTS =====")

    if not results:
        print("No relevant chunks found.")
        return

    for result in results:
        print(f"\nChunk ID: {result['chunk_id']}")
        print(f"Relevance score: {result['score']:.2f}")
        print(f"Retrieved context: {result['text']}")


if __name__ == "__main__":
    main()