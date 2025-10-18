## Assignment 10: Editor‑in‑Chief — Multi‑Tool Synthesis

Concept focus: Bring it all together — prompts, structure, streaming, batch

Scenario
- You're the editor‑in‑chief assembling a weekly briefing.
- Inputs: short blurbs from three reporters; Output: a concise brief with sections, bullet points, and a one‑sentence editorial stance.

Your tasks (in `editor_in_chief.py`)
- Parse reporter blurbs to structured notes (pydantic).
- Generate section drafts in parallel via `.batch()`.
- Stream the final editorial stance to stdout with a callback.
- Return a final JSON bundle with sections and stance.

Run
```bash
python -m assignments.assignment_10_editor_in_chief.editor_in_chief
```


