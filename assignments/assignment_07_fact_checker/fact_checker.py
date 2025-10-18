"""
Assignment 7: Claim-to-Checklist Converter

Write a prompt + chain that turns a claim into a verification checklist.
Return strict JSON with fields: claim, checks[].
"""

from typing import Dict

# TODO: uncomment to implement
# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser


def build_checklist(claim: str) -> Dict:
    """Generate a concise checklist for verifying a claim.

    Implementation guidance in docstring only — reuse patterns from earlier assignments.
    """
    return {
        "claim": claim,
        "checks": [
            {
                "question": "TODO",
                "why_it_matters": "TODO",
                "sources_hint": "TODO",
                "confidence": 0.0,
            }
        ],
    }


def main() -> None:
    out = build_checklist("City X will be carbon-neutral by 2030.")
    print(out)


if __name__ == "__main__":
    main()
