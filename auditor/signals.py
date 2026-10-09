"""Cheap judge-free signals. These are hints recorded next to the data, not labels."""
from __future__ import annotations

_REFUSAL_STARTS = (
    "i can't", "i cannot", "i can not", "i won't", "i will not",
    "i'm sorry", "i am sorry", "sorry,", "i'm unable", "i am unable",
    "i'm not able", "i am not able", "as an ai",
)


def compute(response: str) -> dict:
    text = response.strip()
    folded = text.lower().replace("’", "'")
    return {
        "chars": len(text),
        "question_marks": text.count("?"),
        "ends_with_question": text.endswith("?"),
        "refusal_start": folded.startswith(_REFUSAL_STARTS),
    }
