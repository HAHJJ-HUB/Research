"""Target models served by a local Ollama server."""
from __future__ import annotations

import json
import os
import time
import urllib.request
from dataclasses import dataclass
from typing import List, Optional, Protocol


@dataclass
class ChatResult:
    text: str
    seconds: float
    eval_count: Optional[int] = None


class Target(Protocol):
    model: str

    def digest(self) -> Optional[str]: ...

    def chat(self, system: str, user: str, *, temperature: float, seed: int, num_predict: int) -> ChatResult: ...


def default_host() -> str:
    host = os.environ.get("OLLAMA_HOST", "").strip() or "http://127.0.0.1:11434"
    return host if "://" in host else "http://" + host


def _get(obj, key):
    """Read a field from either a dict or an attribute object (the ollama client has used both)."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def installed_models(client) -> List[dict]:
    out = []
    for m in _get(client.list(), "models") or []:
        out.append({
            "name": _get(m, "model") or _get(m, "name"),
            "digest": _get(m, "digest"),
            "size": _get(m, "size"),
        })
    return out


def server_version(host: str, timeout: float = 5.0) -> str:
    """Ask the server for its version over plain HTTP, bypassing any proxy settings."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(host.rstrip("/") + "/api/version", timeout=timeout) as resp:
            return str(json.load(resp).get("version"))
    except Exception as exc:  # reported to the user, not acted on
        return f"unreachable ({exc.__class__.__name__})"


class OllamaTarget:
    def __init__(self, model: str, host: Optional[str] = None, timeout: float = 900.0, client=None):
        self.model = model
        if client is None:
            import ollama  # imported lazily so the rest of the package works without it
            client = ollama.Client(host=host or default_host(), timeout=timeout)
        self._client = client

    def digest(self) -> Optional[str]:
        want = {self.model} if ":" in self.model else {self.model, self.model + ":latest"}
        for m in installed_models(self._client):
            if m["name"] in want:
                return m["digest"]
        return None

    def chat(self, system: str, user: str, *, temperature: float, seed: int, num_predict: int) -> ChatResult:
        started = time.monotonic()
        resp = self._client.chat(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            options={"temperature": temperature, "seed": seed, "num_predict": num_predict},
        )
        return ChatResult(
            text=_get(_get(resp, "message"), "content") or "",
            seconds=time.monotonic() - started,
            eval_count=_get(resp, "eval_count"),
        )
