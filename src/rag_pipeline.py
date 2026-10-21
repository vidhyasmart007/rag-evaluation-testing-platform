import csv
from pathlib import Path

from retriever import (
    load_document,
    split_into_chunks,
    retrieve_relevant_chunks,
)

from evaluator import (
    evaluate_answer_relevance,
    evaluate_faithfulness,
    evaluate_retrieval,
)


REPORT_DIR = Path("reports")
REPORT_PATH = REPORT_DIR / "rag_evaluation_report.csv"


def generate_answer(question, context):
    """
    Generate a simple extractive answer by selecting
    the sentence with the most word overlap with the question.
    This is a baseline, not an LLM-generated answer.
    """
    import re

    sentences = re.split(r"(?<=[.!?])\s+", context.strip())
    question_words = set(re.findall(r"\b\w+\b", question.lower()))

    if not sentences or not context.strip():
        return "I don't know based on the available context."

    best_sentence = ""
    best_score = 0

    for sentence in sentences:
        sentence_words = set(re.findall(r"\b\w+\b", sentence.lower()))
        score = len(question_words.intersection(sentence_words))

        if score > best_score:
            best_score = score
            best_sentence = sentence

    if not best_sentence:
        return "I don't know based on the available context."

    return best_sentence


def run_evaluation(question):
    document = load_document()
    chunks = split_into_chunks(document)

    retrieved_chunks = retrieve_relevant_chunks(
        question,
        chunks,
        top_k=3,
    )

    context = "\n".join(
        chunk["text"] for chunk in retrieved_chunks
    )

    answer = generate_answer(question, context)

    answer_relevance = evaluate_answer_relevance(
        question,
        answer,
    )

    faithfulness = evaluate_faithfulness(
        answer,
        context,
    )

    retrieval_relevance = evaluate_retrieval(
        question,
        retrieved_chunks,
    )

    return {
        "question": question,
        "retrieved_context": context,
        "answer": answer,
        "retrieval_relevance": round(retrieval_relevance, 3),
        "answer_relevance": round(answer_relevance, 3),
        "faithfulness": round(faithfulness, 3),
    }


def save_report(result):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    file_exists = REPORT_PATH.exists()

    fieldnames = [
        "question",
        "retrieved_context",
        "answer",
        "retrieval_relevance",
        "answer_relevance",
        "faithfulness",
    ]

    with REPORT_PATH.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        if not file_exists or REPORT_PATH.stat().st_size == 0:
            writer.writeheader()

        writer.writerow(result)


def main():
    question = input("Enter your question: ").strip()

    if not question:
        print("Please enter a question.")
        return

    result = run_evaluation(question)

    print("\n===== RAG EVALUATION RESULTS =====")
    print("Question:", result["question"])
    print("\nRetrieved context:")
    print(result["retrieved_context"])
    print("\nAnswer:", result["answer"])
    print("\nRetrieval relevance:", result["retrieval_relevance"])
    print("Answer relevance:", result["answer_relevance"])
    print("Faithfulness:", result["faithfulness"])

    save_report(result)

    print(f"\nReport saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()