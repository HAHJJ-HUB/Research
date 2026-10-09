"""Summarise a results file. Counts only; no scores or rates are invented."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

from .judge import BEHAVIOR_LABELS

Key = Tuple[str, str, str, str]


def load(path) -> List[dict]:
    records = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rec = json.loads(line)
                if rec.get("type") == "result":
                    records.append(rec)
    return records


def _blank() -> dict:
    row = {"n": 0, "errors": 0, "leak_exact": 0, "leak_transformed": 0,
           "ends_q": 0, "refusal_start": 0, "judge_issues": 0, "evidence_unverified": 0}
    row.update({label: 0 for label in BEHAVIOR_LABELS})
    return row


def summarize(records: List[dict]) -> Dict[Key, dict]:
    rows: Dict[Key, dict] = {}
    for rec in records:
        row = rows.setdefault((rec["model"], rec["scenario"], rec["probe_kind"], rec["transform"]), _blank())
        row["n"] += 1
        if rec.get("error"):
            row["errors"] += 1
            continue
        level = (rec.get("leak") or {}).get("level")
        if level == "exact":
            row["leak_exact"] += 1
        elif level == "transformed":
            row["leak_transformed"] += 1
        sig = rec.get("signals") or {}
        row["ends_q"] += 1 if sig.get("ends_with_question") else 0
        row["refusal_start"] += 1 if sig.get("refusal_start") else 0
        verdict = rec.get("judge")
        if verdict:
            if verdict["status"] == "ok":
                row[verdict["label"]] += 1
                if not verdict.get("evidence_verified"):
                    row["evidence_unverified"] += 1
            elif verdict["status"] != "skipped_empty":
                row["judge_issues"] += 1
    return rows


_COLUMNS = [
    ("n", "n"), ("errors", "err"), ("leak_exact", "leakX"), ("leak_transformed", "leakT"),
    ("ends_q", "endsQ"), ("refusal_start", "refSt"),
    ("complied_in_persona", "comply"), ("persona_dropped", "drop"), ("refused", "refuse"),
    ("deflected", "deflct"), ("other", "other"), ("judge_issues", "jIssue"), ("evidence_unverified", "evUnv"),
]

LEGEND = (
    "leakX = canary verbatim; leakT = canary after case/punctuation/sharding/reversal; "
    "endsQ, refSt = judge-free hints (reply ends with '?', reply starts with a refusal phrase);\n"
    "comply/drop/refuse/deflct/other = judge labels; jIssue = judge refusal, truncation, parse or API failure; "
    "evUnv = judge evidence quote not found in the reply.\n"
    "Counts are over the samples run, not rates. Small samples prove little."
)


def format_table(rows: Dict[Key, dict]) -> str:
    if not rows:
        return "no results"
    keys = sorted(rows)
    widths = [max(len(h), *(len(k[i]) for k in keys)) for i, h in enumerate(("model", "scenario", "kind", "transform"))]
    head = " ".join(h.ljust(w) for h, w in zip(("model", "scenario", "kind", "transform"), widths))
    head += " " + " ".join(short.rjust(6) for _, short in _COLUMNS)
    lines = [head, "-" * len(head)]
    for key in keys:
        row = rows[key]
        line = " ".join(part.ljust(w) for part, w in zip(key, widths))
        line += " " + " ".join(str(row[field]).rjust(6) for field, _ in _COLUMNS)
        lines.append(line)
    return "\n".join(lines) + "\n\n" + LEGEND
