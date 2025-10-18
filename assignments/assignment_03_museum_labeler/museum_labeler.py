"""
Assignment 3: Museum Curator Label Generator
Concepts: Pydantic structured outputs, prompt design

Goal
- Convert curator notes into a consistent exhibit label using `with_structured_output`.
"""

from typing import List

# TODO: uncomment real imports when implementing
# from pydantic import BaseModel, Field
# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import ChatPromptTemplate


# class ExhibitLabel(BaseModel):
#     title: str
#     period: str
#     materials: List[str]
#     visitor_takeaway: str
#     tags: List[str]


def build_prompt():
    """Return a ChatPromptTemplate for generating labels.

    TODO:
    - System: act as a curator; output must fill all fields concisely.
    - User: accept `notes` variable.
    """
    return None


def generate_label(notes: str):
    """Return an `ExhibitLabel` parsed directly from the LLM response.

    TODO:
    - Instantiate ChatOpenAI (low temperature)
    - llm_structured = llm.with_structured_output(ExhibitLabel)
    - Invoke chain and return model instance
    """
    return {
        "title": "TODO",
        "period": "TODO",
        "materials": [],
        "visitor_takeaway": "",
        "tags": [],
    }


def main() -> None:
    notes = "Small bronze figure, Hellenistic period; depicts a dancer mid-turn, likely votive."
    print("🏛️ Museum Labeler — Assignment 3")
    label = generate_label(notes)
    print(label)


if __name__ == "__main__":
    main()
