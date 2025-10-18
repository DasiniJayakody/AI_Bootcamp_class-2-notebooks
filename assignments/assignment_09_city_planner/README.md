## Assignment 9: City Planner — Scenario Synthesis

Concept focus: Multi-step chain composition, aggregation, prioritization

Scenario
- You generate proposals for improving a neighborhood across transport, green spaces, and safety.

Your tasks (in `city_planner.py`)
- Create 3 subchains (transport, green, safety) from one base prompt using `.bind`.
- Aggregate their outputs into a prioritized shortlist with rationale and estimated impact score.

Run
```bash
python -m assignments.assignment_09_city_planner.city_planner
```


