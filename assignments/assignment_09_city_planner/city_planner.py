"""
Assignment 9: City Planner — Scenario Synthesis

Design three subchains (transport, green, safety), then aggregate into a
prioritized shortlist with impact scores.
"""

from typing import Dict, List

# TODO: uncomment when implementing
# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser


def plan_neighborhood(neighborhood: str) -> Dict[str, List[Dict]]:
    """Return a dict with keys transport/green/safety and a prioritized shortlist.

    Implementation notes in docstring only; apply earlier patterns.
    """
    return {
        "transport": [{"idea": "TODO", "impact": 3}],
        "green": [{"idea": "TODO", "impact": 2}],
        "safety": [{"idea": "TODO", "impact": 2}],
        "prioritized": [{"idea": "TODO", "impact": 3, "why": "..."}],
    }


def main() -> None:
    print("🏙️ City Planner — Assignment 9")
    out = plan_neighborhood("Riverview District")
    print(out)


if __name__ == "__main__":
    main()
