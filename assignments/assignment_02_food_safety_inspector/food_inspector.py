"""
Assignment 2: AI Food Safety Inspector
"""

import json
import os
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


# ENUMS ---------------------------------------------------------

class ViolationCategory(Enum):
    TEMPERATURE_CONTROL = "Food Temperature Control"
    PERSONAL_HYGIENE = "Personal Hygiene"
    PEST_CONTROL = "Pest Control"
    CROSS_CONTAMINATION = "Cross Contamination"
    FACILITY_MAINTENANCE = "Facility Maintenance"
    UNKNOWN = "Unknown"


class SeverityLevel(Enum):
    CRITICAL = "Critical"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class InspectionPriority(Enum):
    URGENT = "URGENT"
    HIGH = "HIGH"
    ROUTINE = "ROUTINE"
    LOW = "LOW"


# DATA CLASSES --------------------------------------------------

@dataclass
class Violation:
    category: str
    description: str
    severity: str
    evidence: str
    confidence: float


@dataclass
class InspectionReport:
    restaurant_name: str
    overall_risk_score: int
    violations: List[Violation]
    inspection_priority: str
    recommended_actions: List[str]
    follow_up_required: bool


# MAIN CLASS -----------------------------------------------------

class FoodSafetyInspector:

    def __init__(self, model_name="gpt-4o-mini", temperature=0.1):

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("ERROR: OPENAI_API_KEY not set in .env!")

        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=api_key,
        )

        self._setup_chains()

    # ------------------------------------------------------------

    def _setup_chains(self):

        # ANALYSIS PROMPT
        analysis_template = PromptTemplate.from_template("""
You are an AI Food Safety Inspector.
Extract ALL health violations.

STRICT JSON OUTPUT:

{{
  "violations": [
    {{
      "category": "string",
      "description": "string",
      "severity": "Critical|High|Medium|Low",
      "evidence": "exact snippet",
      "confidence": 0.0
    }}
  ]
}}

If no violations, return {{"violations":[]}}.

Review text:
{review_text}

JSON ONLY:
""")

        self.analysis_chain = analysis_template | self.llm | StrOutputParser()

        # RISK PROMPT
        risk_template = PromptTemplate.from_template("""
You are a food safety risk scoring AI.

Violations:
{violations}

Compute score:
Critical=30, High=20, Medium=10, Low=5
If pests or raw meat → +20 bonus

Priority:
>=70 URGENT
>=40 HIGH
else ROUTINE

Return JSON ONLY:
{{
  "risk_score": <int>,
  "priority": "<string>"
}}
""")

        self.risk_chain = risk_template | self.llm | StrOutputParser()

    # ------------------------------------------------------------

    def detect_violations(self, text: str) -> List[Violation]:
        """Run the LLM and parse violations."""
        try:
            raw = self.analysis_chain.invoke({"review_text": text})
            data = json.loads(raw)

            violations = []
            for v in data.get("violations", []):
                violations.append(
                    Violation(
                        category=v.get("category", "Unknown"),
                        description=v.get("description", ""),
                        severity=v.get("severity", "Low"),
                        evidence=v.get("evidence", ""),
                        confidence=float(v.get("confidence", 0.0)),
                    )
                )
            return violations

        except Exception as e:
            print("Violation detection error:", e)
            return []

    # ------------------------------------------------------------

    def calculate_risk_score(self, violations: List[Violation]) -> Tuple[int, str]:
        try:
            vio_json = json.dumps([asdict(v) for v in violations])
            raw = self.risk_chain.invoke({"violations": vio_json})
            data = json.loads(raw)
            return data["risk_score"], data["priority"]

        except Exception as e:
            print("Risk scoring error:", e)
            return 0, "ROUTINE"

    # ------------------------------------------------------------

    def analyze_review(self, text: str, restaurant_name="Unknown") -> InspectionReport:

        violations = self.detect_violations(text)
        risk_score, priority = self.calculate_risk_score(violations)

        recommended_actions = [
            f"Investigate: {v.category}" for v in violations
        ]

        return InspectionReport(
            restaurant_name=restaurant_name,
            overall_risk_score=risk_score,
            violations=violations,
            inspection_priority=priority,
            recommended_actions=recommended_actions,
            follow_up_required=(risk_score >= 40),
        )


# DEMO -----------------------------------------------------------

def _demo_reviews():
    return [
        {
            "restaurant": "Bob's Burgers",
            "text": "Great food but saw a mouse running inside! Chef touched raw chicken then served bread.",
        },
        {
            "restaurant": "Pizza Palace",
            "text": "The bathroom had no soap and meat was sitting unrefrigerated on the counter.",
        },
    ]


def main():
    print("🍽️ FOOD SAFETY INSPECTION SYSTEM — Assignment 2\n")

    insp = FoodSafetyInspector()

    for row in _demo_reviews():
        report = insp.analyze_review(row["text"], row["restaurant"])
        print(json.dumps(asdict(report), indent=2))


if __name__ == "__main__":
    main()
