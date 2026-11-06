"""The exact text the fine-tuned router reads, the same in training and in serving.

No chat template: Hugging Face and Ollama apply templates differently, and a model that
is trained on one text and served another fails silently (it still answers, just
worse). So the text is built here, once, and the server sends it to Ollama in raw mode.

    <system>

    Request:
    <prompt>

    Tier:

The model continues with " S", " M" or " L". Any change to this layout is a new
FORMAT_VERSION, recorded in every bundle so a model is never served with the wrong one.
"""

from __future__ import annotations

from pitwall.data import Pair

FORMAT_VERSION = "pitwall-completion/1"
# The answer the model learns, with its leading space (one token in Qwen's vocabulary).
ANSWERS: dict[str, str] = {"S": " S", "M": " M", "L": " L"}


def render(system: str, user: str) -> str:
    return f"{system}\n\nRequest:\n{user}\n\nTier:"


def render_pair(pair: Pair) -> tuple[str, str]:
    """(prompt text, answer text) for one training pair."""
    by_role = {m["role"]: m["content"] for m in pair.messages}
    return render(by_role["system"], by_role["user"]), ANSWERS[pair.label]
