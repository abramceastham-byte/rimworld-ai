"""A stand-in for a real LLM, so you can test your agent loop offline.

make_fake_model() returns a function with the same shape your loop will
use for a real model: it takes a prompt string and returns a reply string.
Replies come from a fixed list and repeat in order, so runs are reproducible.
"""

import itertools
import json
from collections.abc import Callable, Iterable

DEFAULT_ANSWERS = [
    json.dumps({"actions": [], "time": "hold"}),
    json.dumps({"actions": [], "time": "advance"}),
]


def make_fake_model(answers: Iterable[str] = DEFAULT_ANSWERS) -> Callable[[str], str]:
    answers = list(answers)
    if not answers:
        raise ValueError("make_fake_model needs at least one answer")
    cycle = itertools.cycle(answers)

    def fake_model(prompt: str) -> str:
        return next(cycle)

    return fake_model
