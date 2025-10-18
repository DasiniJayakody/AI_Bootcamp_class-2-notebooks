## Assignment 5: Micro-Learning Coach (Batching)

Concept focus: Runnable.batch for concurrent prompts

Scenario
- You are writing an academic coach that answers many bite-sized questions quickly.

Your tasks (in `batch_coach.py`)
- Build a `chain = llm | StrOutputParser()`.
- Accept a list of questions and answer them concurrently via `.batch()`.
- Print question → answer pairs neatly with indices.

Run
```bash
python -m assignments.assignment_05_batch_coach.batch_coach
```


