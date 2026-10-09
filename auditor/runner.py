"""Build the test matrix and run it against Ollama targets, one request at a time.

Every request is a fresh single-shot: system prompt plus one user message, no history. There is no
multi-turn state, so a model that answers with questions cannot trap the run. The reply is recorded
and the run moves on.
"""
from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

from . import __version__, canary, signals
from .judge import JudgeResult
from .suites import EXTRACTION_PROBES
from .transforms import apply_transform


class RunAborted(RuntimeError):
    pass


@dataclass
class RunConfig:
    models: List[str]
    scenarios: List[dict]
    transforms: List[str]
    include_extraction: bool = True
    samples: int = 1
    temperature: float = 0.8
    seed: int = 1234
    num_predict: int = 256
    hint: bool = True
    judge_budget: int = 100
    max_consecutive_failures: int = 5


@dataclass(frozen=True)
class Case:
    model: str
    scenario: str
    probe_kind: str  # "scenario" | "extraction"
    probe_id: str
    probe_text: str
    transform: str
    sample: int


def build_cases(cfg: RunConfig) -> List[Case]:
    """Models are the outermost loop so each model is loaded once, which matters on small machines."""
    cases: List[Case] = []
    for model in cfg.models:
        for scenario in cfg.scenarios:
            probes = [("scenario", "scenario_probe", scenario["probe"])]
            if cfg.include_extraction:
                probes += [("extraction", p["id"], p["text"]) for p in EXTRACTION_PROBES]
            for kind, probe_id, text in probes:
                for transform in cfg.transforms:
                    for sample in range(cfg.samples):
                        cases.append(Case(model, scenario["name"], kind, probe_id, text, transform, sample))
    return cases


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run(cfg: RunConfig, make_target: Callable[[str], object], judge, out_path: Path,
        log: Callable[[str], None] = print) -> Path:
    cases = build_cases(cfg)
    if judge is not None and len(cases) > cfg.judge_budget:
        raise ValueError(
            f"{len(cases)} judge calls planned but the judge budget is {cfg.judge_budget}; "
            "raise --judge-budget or shrink the matrix")
    scenarios: Dict[str, dict] = {s["name"]: s for s in cfg.scenarios}

    targets = {m: make_target(m) for m in cfg.models}
    digests: Dict[str, Optional[str]] = {}
    for model, target in targets.items():
        try:
            digests[model] = target.digest()
        except Exception as exc:
            raise RunAborted(f"could not list models from Ollama: {type(exc).__name__}: {exc}") from exc
        if digests[model] is None:
            log(f"warning: {model!r} not found in the Ollama model list; calls will probably fail")

    if judge is not None:
        log("checking the judge model with one small request ...")
        judge.ping()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    target_failures = 0
    judge_failures = 0
    with out_path.open("w", encoding="utf-8") as fh:
        meta = {
            "type": "meta",
            "harness_version": __version__,
            "started": _now(),
            "python": platform.python_version(),
            "models": digests,
            "judge": None if judge is None else {"model": judge.model, "effort": judge.effort},
            "config": {
                "scenarios": [s["name"] for s in cfg.scenarios],
                "transforms": cfg.transforms,
                "include_extraction": cfg.include_extraction,
                "samples": cfg.samples,
                "temperature": cfg.temperature,
                "base_seed": cfg.seed,
                "num_predict": cfg.num_predict,
                "transform_hint": cfg.hint,
                "planned_cases": len(cases),
            },
        }
        fh.write(json.dumps(meta, ensure_ascii=False) + "\n")
        fh.flush()

        for n, case in enumerate(cases, 1):
            scenario = scenarios[case.scenario]
            system = scenario["system_prompt"]
            user = apply_transform(case.transform, case.probe_text, hint=cfg.hint)
            seed = cfg.seed + case.sample
            rec = {
                "type": "result", "n": n, "ts": _now(),
                "model": case.model, "digest": digests[case.model],
                "scenario": case.scenario, "probe_kind": case.probe_kind, "probe_id": case.probe_id,
                "transform": case.transform, "sample": case.sample,
                "seed": seed, "temperature": cfg.temperature, "num_predict": cfg.num_predict,
                "system_prompt_sha256": hashlib.sha256(system.encode("utf-8")).hexdigest(),
                "user_message": user,
                "response": None, "seconds": None, "eval_count": None,
                "signals": None, "leak": None, "judge": None, "error": None,
            }
            try:
                res = targets[case.model].chat(
                    system, user, temperature=cfg.temperature, seed=seed, num_predict=cfg.num_predict)
            except Exception as exc:  # one failed call is recorded, repeated failures abort below
                rec["error"] = f"{type(exc).__name__}: {exc}"
                target_failures += 1
            else:
                target_failures = 0
                hit = canary.detect(res.text, scenario["canary_token"])
                rec.update(
                    response=res.text, seconds=round(res.seconds, 2), eval_count=res.eval_count,
                    signals=signals.compute(res.text), leak=hit._asdict())
                if judge is not None:
                    if res.text.strip():
                        verdict = judge.judge(
                            scenario_system=system, probe=case.probe_text,
                            transform=case.transform, reply=res.text)
                    else:
                        verdict = JudgeResult(status="skipped_empty")
                    rec["judge"] = asdict(verdict)
                    judge_failures = judge_failures + 1 if verdict.status == "api_error" else 0

            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            leak = (rec["leak"] or {}).get("level", "-")
            verdict_text = (rec["judge"] or {}).get("label") or (rec["judge"] or {}).get("status") or "-"
            log(f"[{n}/{len(cases)}] {case.model} {case.scenario}/{case.probe_id}/{case.transform}/s{case.sample} "
                f"leak={leak} judge={verdict_text} {rec['seconds'] if rec['seconds'] is not None else 'ERROR'}")
            if target_failures >= cfg.max_consecutive_failures:
                raise RunAborted(
                    f"{target_failures} target calls failed in a row (last: {rec['error']}); "
                    f"partial results kept in {out_path}")
            if judge_failures >= cfg.max_consecutive_failures:
                raise RunAborted(
                    f"{judge_failures} judge calls failed in a row; partial results kept in {out_path}")
    return out_path
