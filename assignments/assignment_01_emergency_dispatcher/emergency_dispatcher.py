"""
Assignment 1: Emergency Dispatcher Triage
Concepts: ChatOpenAI, ChatPromptTemplate, LCEL, StrOutputParser

Scenario
- You are building a tiny helper for a community emergency hotline.
- Given a caller message, classify priority and explain the rationale in one line.

Deliverable
- Implement a minimal chain: prompt | llm | StrOutputParser()
- Output a compact JSON string like: {"priority": "URGENT", "reason": "..."}

Guidance
- Keep temperature low for consistency.
- If uncertain, return UNKNOWN.
- No extra prose around JSON.
"""

import os
from typing import List
from enum import Enum
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()
print("Loaded key:", os.getenv("OPENAI_API_KEY")) #If there is an error : run -> Remove-Item Env:\OPENAI_API_KEY


# TODO: import ChatOpenAI, ChatPromptTemplate, StrOutputParser
# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser


class TriagePriority(Enum):
    LIFE_THREATENING = "LIFE_THREATENING"
    URGENT = "URGENT"
    NON_URGENT = "NON_URGENT"
    UNKNOWN = "UNKNOWN"


class EmergencyDispatcher:
    """Minimal triage assistant driven by a single LangChain pipeline."""

    def __init__(
        self, model_name: str = "gpt-4o-mini", temperature: float = 1.0
    ) -> None:
        # TODO: initialize a ChatOpenAI LLM configured with provided model and temperature
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=os.getenv("OPENAI_API_KEY"),
        )

    def build_prompt(self):
        """Return a ChatPromptTemplate defining system+user roles.

        TODO:
        - Use system instructions to require JSON-only output with keys `priority` and `reason`.
        - Accept a single variable `message` for the caller text.
        - Provide a short rubric for mapping message → priority:
          LIFE_THREATENING (eg. not breathing, severe bleeding),
          URGENT (eg. injury needs care soon),
          NON_URGENT (eg. noise complaint),
          UNKNOWN if insufficient detail.
        """

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a precise emergency dispatcher assistant. "
                    "Classify the caller message and output JSON only in this format: "
                    '{{"priority": "<LIFE_THREATENING|URGENT|NON_URGENT|UNKNOWN>", "reason": "<one-line reason>"}}. '
                    "Use these rules: "
                    "LIFE_THREATENING (e.g., not breathing, severe bleeding), "
                    "URGENT (e.g., injury needing care soon), "
                    "NON_URGENT (e.g., minor issue, complaint), "
                    "UNKNOWN if unclear.",
                ),
                ("user", "Caller message: {message}"),
            ]
        )
        return prompt

    def triage(self, message: str) -> str:
        """Return compact JSON with keys `priority` and `reason`.

        TODO:
        - Compose chain = prompt | llm | StrOutputParser()
        - Invoke with {"message": message}
        - Return the resulting JSON string (do not prettify)
        """

        prompt = self.build_prompt()
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"message": message})


def sample_calls() -> List[str]:
    return [
        "He collapsed, barely breathing, turning blue! We need help now!",
        "Bike crash, bleeding from knee, can someone come soon?",
        "There is a loud party next door again.",
        "Um... something might be wrong but I'm not sure...",
    ]


def main() -> None:
    print("🚨 Emergency Dispatcher Triage - Assignment 1")
    dispatcher = EmergencyDispatcher()

    for msg in sample_calls():
        result = dispatcher.triage(msg)
        print(f"\nCALL: {msg[:70]}...")
        print(f"RESULT: {result}")


if __name__ == "__main__":
    main()
