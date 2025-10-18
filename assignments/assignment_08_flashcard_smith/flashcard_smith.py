"""
Assignment 8: Flashcard Smith

Produce structured flashcards from blurbs using pydantic + batch.
"""

from typing import List

# TODO: uncomment when implementing
# from pydantic import BaseModel, Field
# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import ChatPromptTemplate


# class Flashcard(BaseModel):
#     term: str
#     definition: str
#     difficulty: str


def make_flashcards(blurbs: List[str]):
    """Return a list of Flashcard models generated concurrently.

    Implementation details to mirror Assignment 3 + 5.
    """
    return [
        {"term": "TODO", "definition": "TODO", "difficulty": "easy"} for _ in blurbs
    ]


def main() -> None:
    topics = [
        "Transformer attention",
        "Gradient descent",
        "Overfitting",
    ]
    print("🃏 Flashcard Smith — Assignment 8")
    out = make_flashcards(topics)
    for i, f in enumerate(out, start=1):
        print(f"{i}. {f['term']} — {f['difficulty']}: {f['definition']}")


if __name__ == "__main__":
    main()
