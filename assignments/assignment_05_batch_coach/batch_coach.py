"""
Assignment 5: Micro-Learning Coach
Concepts: Runnable.batch for concurrent inference
"""

from typing import List

# TODO: uncomment when implementing
# from langchain_openai import ChatOpenAI
# from langchain_core.output_parsers import StrOutputParser


def answer_batch(questions: List[str]) -> List[str]:
    """Return answers for each question using `.batch()`.

    TODO:
    - Build `chain = llm | StrOutputParser()`
    - return chain.batch(questions)
    """
    return ["TODO: implement" for _ in questions]


def demo_questions() -> List[str]:
    return [
        "Define overfitting in one sentence.",
        "Explain dropout briefly.",
        "Contrast precision vs recall in one line.",
        "What does temperature do in LLMs?",
    ]


def main() -> None:
    print("🧠 Batch Coach — Assignment 5")
    qs = demo_questions()
    answers = answer_batch(qs)
    for i, (q, a) in enumerate(zip(qs, answers), start=1):
        print(f"\n[{i}] Q: {q}\n    A: {a}")


if __name__ == "__main__":
    main()
