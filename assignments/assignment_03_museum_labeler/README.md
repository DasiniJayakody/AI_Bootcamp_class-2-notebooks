## Assignment 3: Museum Curator Label Generator

Concept focus: Pydantic structured outputs via `llm.with_structured_output`

Scenario
- You assist a museum curator who needs consistent exhibit labels generated from short curator notes.
- Labels should include: `title`, `period`, `materials`, `visitor_takeaway` (1 sentence), and `tags`.

Your tasks (in `museum_labeler.py`)
- Define a `pydantic.BaseModel` for the label structure.
- Build a prompt and wrap the LLM using `with_structured_output` to directly parse into the model.
- Provide a small CLI that prints a generated label for sample notes.

Run
```bash
python -m assignments.assignment_03_museum_labeler.museum_labeler
```


