"""
Assignment 6: Multi‑Voice Moderator

Implement a base prompt once, then create two runtime configurations:
- mentor_chain: temperature 0.2, concise actionable steps
- critic_chain: temperature 0.7, highlights risks, missing considerations

Return a JSON with both perspectives and a short synthesis.
"""

from typing import Dict

# TODO: uncomment to implement
# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser


def advise(topic: str) -> Dict[str, str]:
    """Run two differently-bound chains and combine results.

    Steps expected:
    - Build a ChatPromptTemplate with system instructions and a user slot `{topic}`.
    - Base llm = ChatOpenAI(model="gpt-4o-mini").
    - mentor_chain = (prompt | llm.bind(temperature=0.2) | StrOutputParser())
    - critic_chain = (prompt | llm.bind(temperature=0.7) | StrOutputParser())
    - return {"mentor_advice": ..., "critic_pushback": ..., "synthesis": ...}
    """

    return {
        "mentor_advice": "TODO",
        "critic_pushback": "TODO",
        "synthesis": "TODO",
    }


def main() -> None:
    print("🗣️ Multi‑Voice Moderator — Assignment 6")
    out = advise("launching a student coding club")
    print(out)


if __name__ == "__main__":
    main()
