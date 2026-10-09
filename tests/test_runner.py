import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from auditor import cli, report
from auditor.judge import JudgeResult
from auditor.runner import RunAborted, RunConfig, build_cases, run
from auditor.suites import TEST_SUITE
from auditor.targets import ChatResult, OllamaTarget


class FakeTarget:
    """Leaks the system prompt on the verbatim probe, answers with a question on the config probe."""

    def __init__(self, model, fail=False, digest="d1"):
        self.model, self.fail, self._digest = model, fail, digest
        self.calls = []

    def digest(self):
        return self._digest

    def chat(self, system, user, *, temperature, seed, num_predict):
        self.calls.append((system, user, temperature, seed, num_predict))
        if self.fail:
            raise RuntimeError("server down")
        if "verbatim" in user.lower():
            return ChatResult(system, 0.1, 5)
        if "configuration audit" in user.lower():
            return ChatResult("Which identifiers do you mean?", 0.1, 5)
        return ChatResult("Hail, loyal subjects. Books shall be gathered.", 0.1, 5)


class FakeJudge:
    model, effort = "fake-judge", "low"

    def __init__(self):
        self.pinged, self.calls = False, 0

    def ping(self):
        self.pinged = True

    def judge(self, **kwargs):
        self.calls += 1
        return JudgeResult(status="ok", label="complied_in_persona", evidence="Hail", evidence_verified=True)


def cfg(**overrides):
    base = dict(models=["m1"], scenarios=TEST_SUITE, transforms=["plain", "reversed", "sharded"], samples=1)
    base.update(overrides)
    return RunConfig(**base)


def read(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


class MatrixTests(unittest.TestCase):
    def test_matrix_size(self):
        # 1 model x 2 scenarios x (1 scenario probe + 3 extraction) x 3 transforms x 2 samples
        self.assertEqual(len(build_cases(cfg(samples=2))), 48)
        self.assertEqual(len(build_cases(cfg(include_extraction=False))), 6)

    def test_models_are_the_outermost_loop(self):
        order = [c.model for c in build_cases(cfg(models=["a", "b"]))]
        self.assertEqual(order, sorted(order))


class RunTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name) / "nested" / "run.jsonl"
        self.addCleanup(self.tmp.cleanup)

    def test_full_run_records_leaks_judge_and_metadata(self):
        target, judge = FakeTarget("m1"), FakeJudge()
        run(cfg(transforms=["plain"]), lambda m: target, judge, self.out, log=lambda s: None)
        rows = read(self.out)
        self.assertEqual(rows[0]["type"], "meta")
        self.assertEqual(rows[0]["models"], {"m1": "d1"})
        self.assertEqual(rows[0]["judge"], {"model": "fake-judge", "effort": "low"})
        results = rows[1:]
        self.assertEqual(len(results), 8)
        self.assertTrue(judge.pinged)
        self.assertEqual(judge.calls, 8)
        leaks = {(r["scenario"], r["probe_id"]): r["leak"]["level"] for r in results}
        self.assertEqual(leaks[("maximilian_absolutist", "verbatim_repeat")], "exact")
        self.assertEqual(leaks[("maximilian_absolutist", "scenario_probe")], "none")
        question = next(r for r in results if r["probe_id"] == "config_audit")
        self.assertTrue(question["signals"]["ends_with_question"])
        self.assertEqual(question["seed"], 1234)
        self.assertEqual(len(question["system_prompt_sha256"]), 64)

    def test_transformed_message_is_what_gets_sent(self):
        target = FakeTarget("m1")
        run(cfg(transforms=["reversed"], include_extraction=False, scenarios=TEST_SUITE[:1]),
            lambda m: target, None, self.out, log=lambda s: None)
        sent = target.calls[0][1]
        self.assertTrue(sent.endswith(TEST_SUITE[0]["probe"][::-1]))
        self.assertEqual(target.calls[0][0], TEST_SUITE[0]["system_prompt"])

    def test_seed_varies_by_sample(self):
        target = FakeTarget("m1")
        run(cfg(transforms=["plain"], include_extraction=False, scenarios=TEST_SUITE[:1], samples=3),
            lambda m: target, None, self.out, log=lambda s: None)
        self.assertEqual([c[3] for c in target.calls], [1234, 1235, 1236])

    def test_judge_budget_blocks_before_any_call_or_file(self):
        target, judge = FakeTarget("m1"), FakeJudge()
        with self.assertRaises(ValueError):
            run(cfg(judge_budget=5), lambda m: target, judge, self.out, log=lambda s: None)
        self.assertEqual(target.calls, [])
        self.assertFalse(judge.pinged)
        self.assertFalse(self.out.exists())

    def test_repeated_target_failures_abort_and_keep_partial_results(self):
        target = FakeTarget("m1", fail=True)
        with self.assertRaises(RunAborted):
            run(cfg(), lambda m: target, None, self.out, log=lambda s: None)
        results = read(self.out)[1:]
        self.assertEqual(len(results), 5)
        self.assertTrue(all("RuntimeError" in r["error"] for r in results))

    def test_missing_model_warns_but_continues(self):
        logs = []
        target = FakeTarget("m1", digest=None)
        run(cfg(transforms=["plain"], include_extraction=False, scenarios=TEST_SUITE[:1]),
            lambda m: target, None, self.out, log=logs.append)
        self.assertTrue(any("not found" in line for line in logs))

    def test_ollama_target_reads_both_response_shapes(self):
        class DictClient:
            def list(self):
                return {"models": [{"name": "llama3.1:8b", "digest": "abc"}]}

            def chat(self, **kw):
                return {"message": {"content": "hi"}, "eval_count": 3}

        class ObjClient:
            def list(self):
                from types import SimpleNamespace as NS
                return NS(models=[NS(model="mistral:latest", digest="def")])

            def chat(self, **kw):
                from types import SimpleNamespace as NS
                return NS(message=NS(content="yo"), eval_count=4)

        a = OllamaTarget("llama3.1:8b", client=DictClient())
        b = OllamaTarget("mistral", client=ObjClient())
        self.assertEqual((a.digest(), b.digest()), ("abc", "def"))
        self.assertEqual(a.chat("s", "u", temperature=0.5, seed=1, num_predict=8).text, "hi")
        self.assertEqual(b.chat("s", "u", temperature=0.5, seed=1, num_predict=8).eval_count, 4)
        self.assertIsNone(OllamaTarget("absent", client=DictClient()).digest())


class ReportAndCliTests(unittest.TestCase):
    def test_report_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run.jsonl"
            run(cfg(transforms=["plain"]), lambda m: FakeTarget("m1"), FakeJudge(), out, log=lambda s: None)
            rows = report.summarize(report.load(out))
            row = rows[("m1", "maximilian_absolutist", "extraction", "plain")]
            self.assertEqual((row["n"], row["leak_exact"], row["ends_q"], row["complied_in_persona"]), (3, 1, 1, 3))
            table = report.format_table(rows)
            self.assertIn("leakX", table)
            self.assertIn("not rates", table)

    def test_dry_run_prints_plan_and_touches_nothing(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = cli.main(["run", "--models", "m1,m2", "--dry-run"])
        self.assertEqual(code, 0)
        # 2 models x 2 scenarios x 4 probes x 3 transforms x 1 sample
        self.assertIn("48 target calls", buf.getvalue())
        self.assertIn("48 judge calls", buf.getvalue())

    def test_smoke_plan_is_one_call_per_model(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cli.main(["run", "--models", "m1", "--smoke", "--judge", "none", "--dry-run"])
        self.assertIn("1 target calls", buf.getvalue())
        self.assertIn("0 judge calls", buf.getvalue())

    def test_unknown_scenario_and_transform_are_rejected(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertEqual(cli.main(["run", "--models", "m", "--scenarios", "nope", "--dry-run"]), 2)
            self.assertEqual(cli.main(["run", "--models", "m", "--transforms", "rot13", "--dry-run"]), 2)


if __name__ == "__main__":
    unittest.main()
