"""Shared ChatOpenAI options so every agent honors OPENAI_REASONING_EFFORT."""
import os


def reasoning_kwargs() -> dict[str, object]:
    # Reasoning models are much faster at "low"; set OPENAI_REASONING_EFFORT= (empty) for
    # non-reasoning models, which then use temperature 0 instead.
    effort = os.getenv("OPENAI_REASONING_EFFORT", "low").strip()
    return {"reasoning_effort": effort} if effort else {"temperature": 0}
