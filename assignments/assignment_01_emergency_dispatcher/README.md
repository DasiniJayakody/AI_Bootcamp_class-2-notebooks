## Assignment 1: Emergency Dispatcher Triage

Concept focus: ChatOpenAI basics, ChatPromptTemplate, StrOutputParser

Scenario
- You volunteer at a community emergency hotline. Calls come in with mixed clarity.
- Your job is to build a tiny triage helper that classifies a caller message as one of:
  - LIFE_THREATENING
  - URGENT
  - NON_URGENT
  - UNKNOWN
- The assistant must also extract a one‑line rationale.

Your tasks (in `emergency_dispatcher.py`)
- Implement a `ChatPromptTemplate` with a clear system role and a templated user message.
- Compose a simple LCEL pipeline: `prompt | llm | StrOutputParser()`.
- Return a compact JSON string like: `{ "priority": "URGENT", "reason": "..." }`.

Guardrails
- Be concise. No extra prose around JSON.
- Avoid overconfident claims; if uncertain, return `UNKNOWN`.

Stretch ideas
- Add `temperature` controls and compare outputs.
- Add a simple function that maps caller-provided location cues ("near the old bridge") to a rough area tag used in logging.

How to run
```bash
python -m assignments.assignment_01_emergency_dispatcher.emergency_dispatcher
```


