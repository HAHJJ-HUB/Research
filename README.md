# Local-model audit harness

Tests open-weights models served by a local Ollama server for two things:

1. **Canary leakage.** A fake secret ("canary") is planted in each scenario's system prompt. The harness checks
   every reply for it with plain string matching, including case changes, dropped brackets, sharding and
   reversal. No model is involved in this check.
2. **Behaviour under persona pressure.** Claude labels each reply as `complied_in_persona`, `persona_dropped`,
   `refused`, `deflected` or `other`, and must quote the evidence.

Every request is a fresh single-shot (system prompt plus one user message, no history). A model that answers with
questions cannot trap the run; the reply is recorded as `deflected` and the run moves on.

All canaries and personas are synthetic. Do not put real credentials or production prompts in a suite.

## Setup (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1        # if blocked: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
pip install -r requirements.txt
$env:ANTHROPIC_API_KEY = "<your key>"   # this session only. Never paste a key into chat or commit it.
```

## First run, in this order

```powershell
# 1. Real facts about your machine: versions, Ollama server, installed models + digests, free disk.
python -m auditor preflight --models <name from `ollama list`> --check-judge

# 2. See the plan and judge-call count without running anything.
python -m auditor run --models <model> --dry-run

# 3. One local call, no Claude involved. Confirms Ollama wiring and shows real speed on your laptop.
python -m auditor run --models <model> --smoke --judge none

# 4. The same one call, judged by Claude (costs a few API calls).
python -m auditor run --models <model> --smoke

# 5. Summarise.
python -m auditor report results\<file>.jsonl
```

`--check-judge` makes one tiny API call so a bad key or a retired model ID fails before any long local run.
A full default run is 24 calls per model per sample (2 scenarios x 4 probes x 3 transforms). On an 8 GB laptop
expect it to be slow; start with `--smoke`, then `--transforms plain --samples 1`.

## Judge model

Default is `claude-sonnet-5-5`. Override with `--judge-model` or the `AUDITOR_JUDGE_MODEL` environment variable.
`claude-3-5-sonnet-20241022` is a retired model (retired Oct 28, 2025) and will fail.

## Layout

| Path | Purpose |
|---|---|
| `auditor/suites.py` | Your two scenarios verbatim, plus generic extraction probes |
| `auditor/transforms.py` | `plain`, `reversed` (character inversion), `sharded` (4-character fragments joined by `\|`) |
| `auditor/canary.py` | Deterministic leak detection |
| `auditor/targets.py` | Ollama client wrapper |
| `auditor/judge.py` | Claude judge, refusal-aware, injection-resistant prompt |
| `auditor/runner.py` | Matrix, execution, JSONL output |
| `auditor/report.py` | Counts per model / scenario / probe kind / transform |
| `tests/` | `python -m unittest discover -s tests` |

Results go to `results/*.jsonl` (git-ignored). Each line records the model digest, seed, exact message sent,
raw reply, judge-free signals, leak result and judge verdict.

## Known limits

- **Not yet run against a live Ollama server or the live Anthropic API.** The tests use stubs and a local fake
  HTTP server driven by the real client libraries. They check wiring, not real model behaviour.
- **Single-shot only.** There is no scripted multi-turn mode yet.
- **The judge is not deterministic.** Sonnet 5.5 does not accept custom sampling settings. Spot-check verdicts;
  `evUnv` in the report counts verdicts whose quoted evidence is not actually in the reply.
- **The judge can refuse.** Sonnet 5.5 has safety classifiers (including `cyber`). A refused judgement is recorded
  as `refusal` with its category and counted in `jIssue`. Server-side fallback is not wired in. For legitimate
  security work, Anthropic points to the Cyber Verification Program:
  https://support.claude.com/en/articles/14604842-real-time-cyber-safeguards-on-claude-opus-and-sonnet
- **Design choices that are mine, not yours, and are easy to change:** the three extraction probes, the decoding hint
  prepended to reversed and sharded messages (`--no-transform-hint` removes it), temperature 0.8, `num_predict` 256,
  base seed 1234, judge budget 100, judge effort `low`.
- **`num_predict` truncates replies.** A leak after the cutoff is missed. Raise it if speed allows.
- **Ideological bias has no label yet.** It needs a rubric you define.
- **Small models only.** Results from 1B to 8B models describe those models, not larger ones.
