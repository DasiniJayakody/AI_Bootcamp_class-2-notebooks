"""
Assignment 4: Live Performance Poet
Concepts: Streaming tokens, custom callback handler
"""

from typing import Any

# TODO: uncomment when implementing
# from langchain_core.callbacks import BaseCallbackHandler
# from langchain_openai import ChatOpenAI
# from langchain_core.output_parsers import StrOutputParser


class PrintHandler:  # TODO: subclass BaseCallbackHandler
    """TODO: Stream tokens as they arrive.

    - Print tokens without newlines.
    - Optionally add a subtle flush or spacing.
    """

    # def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
    #     print(token, end="")
    pass


def stream_poem(theme: str, lines: int = 4) -> None:
    """Stream a short poem for a given theme.

    TODO:
    - Instantiate ChatOpenAI with streaming=True and callbacks=[PrintHandler()].
    - Compose llm | StrOutputParser() and invoke with an instruction.
    """
    print("[TODO streaming output]")


def main() -> None:
    print("🎤 Streaming Poet — Assignment 4")
    stream_poem("rain-soaked city", lines=5)


if __name__ == "__main__":
    main()
