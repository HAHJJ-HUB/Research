"""Command line: preflight, run, report."""
from __future__ import annotations

import argparse
import importlib.metadata as metadata
import os
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import report as report_mod
from .judge import DEFAULT_JUDGE_MODEL, ClaudeJudge, JudgeFatalError
from .runner import RunAborted, RunConfig, build_cases, run
from .suites import TEST_SUITE, load_suite_file, validate_suite
from .targets import OllamaTarget, default_host, installed_models, server_version
from .transforms import TRANSFORMS

EFFORTS = ("none", "low", "medium", "high", "xhigh", "max")
INSTALL_HINT = "Missing a dependency. Run: pip install -r requirements.txt"


def _csv(value: str):
    return [part.strip() for part in value.split(",") if part.strip()]


def _pkg_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "NOT INSTALLED"


def _make_judge(args):
    return ClaudeJudge(args.judge_model, effort=None if args.judge_effort == "none" else args.judge_effort)


def cmd_preflight(args) -> int:
    ok = True
    host = args.host or default_host()
    print(f"python          {platform.python_version()} on {platform.system()}")
    print(f"ollama (pkg)    {_pkg_version('ollama')}")
    print(f"anthropic (pkg) {_pkg_version('anthropic')}")
    print(f"ollama host     {host}")
    version = server_version(host)
    print(f"ollama server   {version}")
    if version.startswith("unreachable"):
        ok = False
    free_gb = shutil.disk_usage(Path.home()).free / 1e9
    print(f"free disk       {free_gb:.1f} GB on the home drive (models may live elsewhere if OLLAMA_MODELS is set)")
    key = "set" if os.environ.get("ANTHROPIC_API_KEY") else "not set (the SDK may still find another credential source)"
    print(f"ANTHROPIC_API_KEY {key}")

    try:
        import ollama
        models = installed_models(ollama.Client(host=host))
        print("installed models:")
        for m in models:
            size = f"{m['size'] / 1e9:.1f} GB" if isinstance(m.get("size"), (int, float)) else "?"
            print(f"  {m['name']}  {size}  digest={m['digest']}")
        names = {m["name"] for m in models}
        for wanted in _csv(args.models):
            have = wanted in names or (":" not in wanted and wanted + ":latest" in names)
            print(f"  requested {wanted}: {'present' if have else 'MISSING'}")
            ok = ok and have
    except ImportError:
        print(INSTALL_HINT)
        ok = False
    except Exception as exc:
        print(f"could not list models: {type(exc).__name__}: {exc}")
        ok = False

    if args.check_judge:
        try:
            _make_judge(args).ping()
            print(f"judge           {args.judge_model}: ok")
        except ImportError:
            print(INSTALL_HINT)
            ok = False
        except JudgeFatalError as exc:
            print(f"judge           {args.judge_model}: FAILED: {exc}")
            ok = False
    return 0 if ok else 1


def cmd_run(args) -> int:
    suite = load_suite_file(args.suite_file) if args.suite_file else TEST_SUITE
    validate_suite(suite)
    if args.scenarios:
        wanted = _csv(args.scenarios)
        unknown = [w for w in wanted if w not in {s["name"] for s in suite}]
        if unknown:
            print(f"unknown scenarios: {unknown}")
            return 2
        suite = [s for s in suite if s["name"] in wanted]
    transforms = _csv(args.transforms)
    bad = [t for t in transforms if t not in TRANSFORMS]
    if bad:
        print(f"unknown transforms {bad}; choose from {sorted(TRANSFORMS)}")
        return 2
    cfg = RunConfig(
        models=_csv(args.models), scenarios=suite, transforms=transforms,
        include_extraction=not args.no_extraction, samples=args.samples,
        temperature=args.temperature, seed=args.seed, num_predict=args.num_predict,
        hint=not args.no_transform_hint, judge_budget=args.judge_budget)
    if args.smoke:
        cfg.scenarios, cfg.transforms, cfg.include_extraction, cfg.samples = suite[:1], ["plain"], False, 1

    cases = build_cases(cfg)
    use_judge = args.judge == "claude"
    print(f"plan: {len(cases)} target calls ({', '.join(cfg.models)}), "
          f"{len(cases) if use_judge else 0} judge calls, num_predict={cfg.num_predict}, temperature={cfg.temperature}")
    if args.dry_run:
        return 0

    out = Path(args.out) if args.out else Path("results") / f"run-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}.jsonl"
    try:
        judge = _make_judge(args) if use_judge else None
        run(cfg, lambda m: OllamaTarget(m, host=args.host, timeout=args.timeout), judge, out)
    except ImportError:
        print(INSTALL_HINT)
        return 1
    except ValueError as exc:
        print(f"error: {exc}")
        return 2
    except (RunAborted, JudgeFatalError) as exc:
        print(f"error: {exc}")
        return 1
    except KeyboardInterrupt:
        print(f"interrupted; partial results are in {out}")
        return 130
    print(f"done: {out}\nsummarise with: python -m auditor report {out}")
    return 0


def cmd_report(args) -> int:
    print(report_mod.format_table(report_mod.summarize(report_mod.load(args.path))))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m auditor")
    sub = parser.add_subparsers(dest="cmd", required=True)

    pre = sub.add_parser("preflight", help="print real environment facts; optionally check the judge model")
    pre.add_argument("--models", default="", help="comma-separated Ollama model names to check for")
    pre.add_argument("--host")
    pre.add_argument("--check-judge", action="store_true", help="make one tiny API call (costs a fraction of a cent)")
    pre.add_argument("--judge-model", default=os.environ.get("AUDITOR_JUDGE_MODEL", DEFAULT_JUDGE_MODEL))
    pre.add_argument("--judge-effort", choices=EFFORTS, default="low")
    pre.set_defaults(func=cmd_preflight)

    r = sub.add_parser("run", help="run the matrix against local Ollama models")
    r.add_argument("--models", required=True, help="comma-separated Ollama model names")
    r.add_argument("--judge", choices=("claude", "none"), default="claude")
    r.add_argument("--judge-model", default=os.environ.get("AUDITOR_JUDGE_MODEL", DEFAULT_JUDGE_MODEL))
    r.add_argument("--judge-effort", choices=EFFORTS, default="low")
    r.add_argument("--judge-budget", type=int, default=100, help="max judge calls allowed for this run")
    r.add_argument("--transforms", default="plain,reversed,sharded")
    r.add_argument("--scenarios", default="", help="comma-separated scenario names (default: all)")
    r.add_argument("--no-extraction", action="store_true", help="skip the generic extraction probes")
    r.add_argument("--no-transform-hint", action="store_true", help="send transformed text with no decoding hint")
    r.add_argument("--samples", type=int, default=1)
    r.add_argument("--temperature", type=float, default=0.8)
    r.add_argument("--seed", type=int, default=1234, help="base seed; sample i uses seed+i")
    r.add_argument("--num-predict", type=int, default=256, help="max tokens generated per reply")
    r.add_argument("--timeout", type=float, default=900.0, help="seconds to wait for one Ollama reply")
    r.add_argument("--host")
    r.add_argument("--suite-file", help="JSON file of scenarios instead of the built-in suite")
    r.add_argument("--smoke", action="store_true", help="first scenario, plain transform, scenario probe only, 1 sample")
    r.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    r.add_argument("--out")
    r.set_defaults(func=cmd_run)

    rep = sub.add_parser("report", help="summarise a results .jsonl file")
    rep.add_argument("path")
    rep.set_defaults(func=cmd_report)
    return parser


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")  # Windows consoles choke on some characters
    args = build_parser().parse_args(argv)
    return args.func(args)
