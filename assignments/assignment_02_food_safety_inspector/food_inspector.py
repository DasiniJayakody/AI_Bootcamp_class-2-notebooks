"""
Assignment 2: AI Food Safety Inspector
Zero-Shot Prompting with Structured Outputs

Your mission: Analyze restaurant reviews and complaints to detect health violations
using only clear instructions — no training examples needed!
"""

import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

# TODO: Uncomment real imports when implementing
# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import StrOutputParser


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


@dataclass
class Violation:
    """Structured violation data"""

    category: str
    description: str
    severity: str
    evidence: str
    confidence: float


@dataclass
class InspectionReport:
    """Complete inspection analysis"""

    restaurant_name: str
    overall_risk_score: int
    violations: List[Violation]
    inspection_priority: str
    recommended_actions: List[str]
    follow_up_required: bool


class FoodSafetyInspector:
    """
    AI-powered food safety analyzer using zero-shot structured prompting.
    """

    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.1):
        """Initialize with LLM for consistent violation detection."""
        # TODO: Replace with a real LLM once ready
        # self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.llm = None
        self.analysis_chain = None
        self.risk_chain = None
        self._setup_chains()

    def _setup_chains(self) -> None:
        """
        TODO #1: Create zero-shot prompts for violation detection and risk assessment.

        Create TWO chains:
        1. analysis_chain: Detects violations and extracts evidence
        2. risk_chain: Calculates risk scores based on violations

        Requirements:
        - Must output valid JSON
        - Include all violation categories
        - Extract specific evidence quotes
        - Generate confidence scores
        """

        # TODO: Create violation detection prompt
        # analysis_template = PromptTemplate.from_template(
        #     """You are a food safety inspector AI. Analyze the following text for health code violations.
        # [Instructions here]
        # Text to analyze: {review_text}
        # Output JSON:"""
        # )

        # TODO: Create risk assessment prompt
        # risk_template = PromptTemplate.from_template(
        #     """Based on these violations, calculate an overall risk score (0-100).
        # [Scoring criteria here]
        # Violations: {violations}
        # Risk Assessment:"""
        # )

        # TODO: Set up the chains using LCEL
        self.analysis_chain = None
        self.risk_chain = None

    def detect_violations(self, text: str) -> List[Violation]:
        """
        TODO #2: Detect health violations from text input.

        Args:
            text: Review, complaint, or social media post

        Returns:
            List of Violation objects with evidence
        """

        try:
            # TODO: Use analysis_chain to detect violations and parse JSON
            # raw = self.analysis_chain.invoke({"review_text": text})
            # data = json.loads(raw)
            data = {"violations": []}
            return [
                Violation(
                    category=v.get("category", ViolationCategory.UNKNOWN.value),
                    description=v.get("description", ""),
                    severity=v.get("severity", SeverityLevel.LOW.value),
                    evidence=v.get("evidence", ""),
                    confidence=float(v.get("confidence", 0.0)),
                )
                for v in data.get("violations", [])
            ]
        except Exception as e:
            print(f"Error detecting violations: {e}")
            return []

    def calculate_risk_score(self, violations: List[Violation]) -> Tuple[int, str]:
        """
        TODO #3: Calculate overall risk score and determine inspection priority.

        Consider: severity levels, count, and category diversity.
        """

        # TODO: Replace with real scoring logic
        risk_score = min(100, len(violations) * 10)
        priority = (
            InspectionPriority.URGENT.value
            if risk_score >= 70
            else (
                InspectionPriority.HIGH.value
                if risk_score >= 40
                else InspectionPriority.ROUTINE.value
            )
        )
        return risk_score, priority

    def analyze_review(
        self, text: str, restaurant_name: str = "Unknown"
    ) -> InspectionReport:
        """
        TODO #4: Complete analysis pipeline for a single review.
        """

        violations = self.detect_violations(text)
        risk_score, priority = self.calculate_risk_score(violations)
        recommendations: List[str] = []  # TODO: derive from violations
        follow_up = risk_score >= 40

        return InspectionReport(
            restaurant_name=restaurant_name,
            overall_risk_score=risk_score,
            violations=violations,
            inspection_priority=priority,
            recommended_actions=recommendations,
            follow_up_required=follow_up,
        )

    def batch_analyze(self, reviews: List[Dict[str, str]]) -> InspectionReport:
        """
        TODO #5: Analyze multiple reviews for the same restaurant.
        """

        all_violations: List[Violation] = []
        for r in reviews:
            all_violations.extend(self.detect_violations(r.get("text", "")))

        risk_score, priority = self.calculate_risk_score(all_violations)

        return InspectionReport(
            restaurant_name="Unknown",
            overall_risk_score=risk_score,
            violations=all_violations,
            inspection_priority=priority,
            recommended_actions=[],
            follow_up_required=risk_score >= 40,
        )


def _demo_reviews() -> List[Dict[str, str]]:
    return [
        {
            "restaurant": "Bob's Burgers",
            "text": "Great food but saw a mouse run across the dining room! Also, the chef wasn't wearing gloves while handling raw chicken.",
        },
        {
            "restaurant": "Pizza Palace",
            "text": "Bathroom had no soap, and meat sat on the counter unrefrigerated 😷",
        },
    ]


def main() -> None:
    print("🍽️ FOOD SAFETY INSPECTION SYSTEM — Assignment 2")
    insp = FoodSafetyInspector()
    for row in _demo_reviews():
        report = insp.analyze_review(row["text"], row["restaurant"])
        print(json.dumps(asdict(report), indent=2, default=str))


if __name__ == "__main__":
    main()
