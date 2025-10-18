## Assignment 4: Live Performance Poet

Concept focus: Streaming tokens with a custom callback handler

Scenario
- You're building a live poetry performance app. The poem should stream word-by-word to the audience.

Your tasks (in `streaming_poet.py`)
- Implement a `BaseCallbackHandler` that streams tokens to stdout with subtle formatting.
- Create a streaming `ChatOpenAI` and compose `llm | StrOutputParser()`.
- Provide a function to stream a poem with a requested vibe and length.

Run
```bash
python -m assignments.assignment_04_streaming_poet.streaming_poet
```


