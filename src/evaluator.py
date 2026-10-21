import re


def normalize_text(text):
    """Normalize text for basic comparison."""
    return re.findall(r"\b\w+\b", text.lower())


def evaluate_answer_relevance(question, answer):
    """
    Basic lexical check: does the answer share meaningful
    words with the question?
    """
    question_words = set(normalize_text(question))
    answer_words = set(normalize_text(answer))

    # Remove common words that don't indicate much relevance.
    stop_words = {
        "what", "is", "are", "the", "a", "an",
        "does", "do", "how", "which", "who",
        "in", "on", "of", "to", "for", "and"
    }

    question_words -= stop_words
    answer_words -= stop_words

    if not question_words or not answer_words:
        return 0.0

    overlap = question_words.intersection(answer_words)

    return len(overlap) / len(question_words)


def evaluate_faithfulness(answer, context):
    """
    Simple baseline check for whether answer terms appear
    in the retrieved context.

    This is NOT a semantic faithfulness metric.
    """
    answer_words = set(normalize_text(answer))
    context_words = set(normalize_text(context))

    # Ignore very short words for this basic check.
    answer_words = {
        word for word in answer_words
        if len(word) > 2
    }

    if not answer_words:
        return 0.0

    supported_words = answer_words.intersection(context_words)

    return len(supported_words) / len(answer_words)


def evaluate_retrieval(question, retrieved_chunks):
    """Return the highest lexical relevance score."""
    if not retrieved_chunks:
        return 0.0

    return max(chunk["score"] for chunk in retrieved_chunks)