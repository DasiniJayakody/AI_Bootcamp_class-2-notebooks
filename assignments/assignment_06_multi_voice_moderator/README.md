## Assignment 6: Multi‑Voice Moderator

Concept focus: Swapping configs at runtime (`.bind`, `.with_config`), multi-role prompts

Scenario
- You are moderating a fast chat where two personas (Mentor and Critic) offer guidance.
- You need to run the same base chain with different runtime configs and merge results.

Your tasks (in `multi_voice_moderator.py`)
- Build a system + user prompt that accepts `topic`.
- Use `.bind()` to create `mentor_chain` (low temperature) and `critic_chain` (higher temperature).
- Invoke both and aggregate into a single JSON object with `mentor_advice` and `critic_pushback`.

Run
```bash
python -m assignments.assignment_06_multi_voice_moderator.multi_voice_moderator
```


