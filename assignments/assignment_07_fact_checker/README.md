## Assignment 7: Claim-to-Checklist Converter

Concept focus: Prompt engineering + structured outputs + confidence fields

Scenario
- You help journalists quickly turn a claim into a checklist of facts to verify.

Your tasks (in `fact_checker.py`)
- Produce a JSON object with: `claim`, `checks[]`, each check has `question`, `why_it_matters`, `sources_hint`, `confidence`.
- Keep the system strict about JSON and concise language.

Run
```bash
python -m assignments.assignment_07_fact_checker.fact_checker
```


