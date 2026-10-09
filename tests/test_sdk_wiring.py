"""Points the REAL anthropic and ollama client libraries at a throwaway local HTTP server.

This checks our wiring against the installed SDK versions (request bodies, exception classes, response
parsing). It does not prove anything about the real Anthropic API or a real Ollama server. Skipped when
the SDKs are not installed.
"""
import json
import os
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

try:
    import anthropic
    import ollama
    HAVE_SDKS = True
except ImportError:
    HAVE_SDKS = False

from auditor.judge import ClaudeJudge, JudgeFatalError
from auditor.targets import OllamaTarget, installed_models, server_version

STATE = {"mode": "ok", "requests": []}

MESSAGE_OK = {
    "id": "msg_test", "type": "message", "role": "assistant", "model": "claude-sonnet-5-5",
    "content": [{"type": "text", "text": '{"label": "refused", "evidence": "I cannot"}'}],
    "stop_reason": "end_turn", "stop_sequence": None,
    "usage": {"input_tokens": 10, "output_tokens": 5},
}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/version":
            self._send(200, {"version": "0.0.0-test"})
        elif self.path == "/api/tags":
            self._send(200, {"models": [{
                "name": "llama3.1:8b", "model": "llama3.1:8b", "modified_at": "2024-01-01T00:00:00Z",
                "size": 4700000000, "digest": "abc123", "details": {}}]})
        else:
            self._send(404, {"error": "unknown"})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        STATE["requests"].append((self.path, body))
        if self.path == "/api/chat":
            self._send(200, {"model": body.get("model"), "created_at": "2024-01-01T00:00:00Z",
                             "message": {"role": "assistant", "content": "Hail, subjects."},
                             "done": True, "eval_count": 3})
        elif self.path == "/v1/messages":
            if STATE["mode"] == "404":
                self._send(404, {"type": "error", "error": {"type": "not_found_error", "message": "model: retired"}})
            elif STATE["mode"] == "refusal":
                self._send(200, {**MESSAGE_OK, "content": [], "stop_reason": "refusal",
                                 "stop_details": {"type": "refusal", "category": "cyber", "explanation": "x"}})
            else:
                self._send(200, MESSAGE_OK)
        else:
            self._send(404, {"error": "unknown"})


@unittest.skipUnless(HAVE_SDKS, "anthropic and ollama are not installed")
class SdkWiringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["NO_PROXY"] = "127.0.0.1,localhost"
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        STATE["mode"], STATE["requests"] = "ok", []

    def judge(self, **kwargs):
        client = anthropic.Anthropic(api_key="test-key", base_url=self.url, max_retries=0)
        return ClaudeJudge(client=client, **kwargs)

    def test_judge_request_body_and_parsed_verdict(self):
        result = self.judge().judge(scenario_system="s", probe="p", transform="plain", reply="I cannot do that")
        self.assertEqual((result.status, result.label, result.evidence_verified), ("ok", "refused", True))
        path, body = STATE["requests"][-1]
        self.assertEqual(path, "/v1/messages")
        self.assertEqual(body["model"], "claude-sonnet-5-5")
        self.assertEqual(body["output_config"], {"effort": "low"})
        for forbidden in ("temperature", "top_p", "top_k", "thinking"):
            self.assertNotIn(forbidden, body)

    def test_real_not_found_error_is_fatal(self):
        STATE["mode"] = "404"
        with self.assertRaises(JudgeFatalError):
            self.judge(model="claude-3-5-sonnet-20241022").ping()

    def test_real_refusal_response_is_recorded(self):
        STATE["mode"] = "refusal"
        result = self.judge().judge(scenario_system="s", probe="p", transform="plain", reply="r")
        self.assertEqual((result.status, result.refusal_category), ("refusal", "cyber"))

    def test_ollama_chat_request_and_parsing(self):
        target = OllamaTarget("llama3.1:8b", host=self.url, timeout=5.0)
        res = target.chat("sys", "usr", temperature=0.5, seed=7, num_predict=32)
        self.assertEqual((res.text, res.eval_count), ("Hail, subjects.", 3))
        path, body = STATE["requests"][-1]
        self.assertEqual(path, "/api/chat")
        self.assertEqual(body["messages"], [{"role": "system", "content": "sys"}, {"role": "user", "content": "usr"}])
        self.assertEqual(body["options"], {"temperature": 0.5, "seed": 7, "num_predict": 32})
        self.assertFalse(body.get("stream"))

    def test_ollama_digest_listing_and_version(self):
        target = OllamaTarget("llama3.1:8b", host=self.url, timeout=5.0)
        self.assertEqual(target.digest(), "abc123")
        self.assertIsNone(OllamaTarget("absent", host=self.url, timeout=5.0).digest())
        listed = installed_models(ollama.Client(host=self.url))
        self.assertEqual((listed[0]["name"], listed[0]["size"]), ("llama3.1:8b", 4700000000))
        self.assertEqual(server_version(self.url), "0.0.0-test")


if __name__ == "__main__":
    unittest.main()
