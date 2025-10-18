## Assignment 2: AI Food Safety Inspector

Concept focus: Zero-shot prompting, PromptTemplate, JSON outputs, multi-step chains

Scenario
- City health dept triages public reviews to spot possible violations quickly.
- Build an analyzer that finds violations, extracts evidence snippets, and scores risk.

Your tasks (in `food_inspector.py`)
- Create two prompts: analysis (detect violations) and risk (score + priority).
- Compose chains and parse JSON into structured Python objects.
- Provide `analyze_review`, `batch_analyze`, and a simple CLI tester.

Hints
- Keep responses machine-parseable JSON. Avoid prose.
- Include confidence scores and handle ambiguity.

Run
```bash
python -m assignments.assignment_02_food_safety_inspector.food_inspector
```


