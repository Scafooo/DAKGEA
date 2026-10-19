"""LLM clients for Luffy-L, with a persistent prompt->response cache.

Every call goes through ``CachedLLMClient``:

* the cache key is sha256(model_id, decoding params, seed, system, user);
* on a hit the stored response is returned and no request is sent;
* on a miss the backend is called and the response is stored *before* returning.

Because the key includes the experiment seed, runs with the same seed reuse
responses (e.g. af=0.5 replays the prefix of af=1.0, and the ``no_cache``
ablation reuses the prompts of ``full``), while different seeds draw
independent generations, as the paper's 5-run protocol requires.

Backends:
  bedrock  - AWS Bedrock Converse API (Llama 3.3 70B in the paper)
  replay   - cache only; a miss raises ``CacheMiss`` (used to regenerate the
             datasets deterministically from a shipped cache, zero API calls)
  fake     - offline placeholder answers, to dry-run the pipeline end to end
             (use a separate cache_path: fake answers must never mix with real ones)
  mock     - user-supplied function, for unit tests
"""

from __future__ import annotations

import hashlib
import json
import random
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from src.logger import get_logger

logger = get_logger(__name__)


class CacheMiss(RuntimeError):
    """Raised by the replay backend when a prompt is not in the cache."""


@dataclass
class LLMResponse:
    text: str
    from_cache: bool
    latency_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    stop_reason: str = ""


# ----------------------------------------------------------------------
# Persistent cache (SQLite, safe for several processes sharing one file)
# ----------------------------------------------------------------------
class ResponseCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), timeout=120, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=120000")
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS calls (
                   key TEXT PRIMARY KEY,
                   model_id TEXT, seed INTEGER, params TEXT,
                   system TEXT, user TEXT, response TEXT,
                   latency_s REAL, input_tokens INTEGER, output_tokens INTEGER,
                   created REAL)"""
        )
        self._conn.commit()

    def get(self, key: str) -> Optional[str]:
        with self._lock:
            row = self._conn.execute("SELECT response FROM calls WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def put(self, key: str, *, model_id: str, seed: int, params: dict, system: str,
            user: str, response: LLMResponse) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO calls VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (key, model_id, seed, json.dumps(params, sort_keys=True), system, user,
                 response.text, response.latency_s, response.input_tokens,
                 response.output_tokens, time.time()),
            )
            self._conn.commit()


def cache_key(model_id: str, params: dict, seed: int, system: str, user: str) -> str:
    payload = json.dumps(
        {"model_id": model_id, "params": params, "seed": seed, "system": system, "user": user},
        sort_keys=True, ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------
# Backends
# ----------------------------------------------------------------------
class _RateLimiter:
    def __init__(self, rps: float):
        self.interval = 1.0 / rps if rps and rps > 0 else 0.0
        self._next = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        if not self.interval:
            return
        with self._lock:
            now = time.monotonic()
            wait = self._next - now
            self._next = max(now, self._next) + self.interval
        if wait > 0:
            time.sleep(wait)


class BedrockBackend:
    """AWS Bedrock Converse API backend (model-agnostic)."""

    RETRYABLE = {
        "ThrottlingException", "ServiceUnavailableException", "ModelTimeoutException",
        "InternalServerException", "ModelNotReadyException", "TooManyRequestsException",
    }

    def __init__(self, model_id: str, region: str, params: dict,
                 requests_per_second: float = 2.0, max_retries: int = 8):
        import boto3  # imported lazily so tests do not need boto3
        from botocore.config import Config as BotoConfig

        self.model_id = model_id
        self.params = params
        self.max_retries = max_retries
        self.rate = _RateLimiter(requests_per_second)
        self.client = boto3.client(
            "bedrock-runtime", region_name=region,
            config=BotoConfig(read_timeout=300, retries={"max_attempts": 1}),
        )

    def __call__(self, system: str, user: str) -> LLMResponse:
        from botocore.exceptions import ClientError

        inference = {
            "maxTokens": int(self.params["max_tokens"]),
            "temperature": float(self.params["temperature"]),
            "topP": float(self.params["top_p"]),
        }
        for attempt in range(self.max_retries + 1):
            self.rate.wait()
            t0 = time.monotonic()
            try:
                out = self.client.converse(
                    modelId=self.model_id,
                    system=[{"text": system}],
                    messages=[{"role": "user", "content": [{"text": user}]}],
                    inferenceConfig=inference,
                )
                text = "".join(
                    block.get("text", "") for block in out["output"]["message"]["content"]
                )
                usage = out.get("usage", {})
                return LLMResponse(
                    text=text, from_cache=False, latency_s=time.monotonic() - t0,
                    input_tokens=int(usage.get("inputTokens", 0)),
                    output_tokens=int(usage.get("outputTokens", 0)),
                    stop_reason=str(out.get("stopReason", "")),
                )
            except ClientError as exc:
                code = exc.response.get("Error", {}).get("Code", "")
                if code not in self.RETRYABLE or attempt == self.max_retries:
                    raise
                delay = min(60.0, 2.0 ** attempt) * (0.5 + random.random())
                logger.warning("[LLM] %s (attempt %d/%d), retrying in %.1fs",
                               code, attempt + 1, self.max_retries, delay)
                time.sleep(delay)
        raise RuntimeError("unreachable")


class ReplayBackend:
    def __call__(self, system: str, user: str) -> LLMResponse:
        raise CacheMiss(
            "Prompt not found in the LLM cache (provider=replay). The cache does not "
            "match this run: check seed, model params, dataset and code version."
        )


class FakeBackend:
    """Offline backend that answers every prompt with well-formed placeholder values.

    Used to dry-run the whole pipeline (runner, writer, output folders) without
    Bedrock: values look like "FAKE-c<call>-<id>-<KG>-<attribute>-<i>".
    Never use its output as experimental data.
    """

    def __init__(self):
        self.calls = 0

    def __call__(self, system: str, user: str) -> LLMResponse:
        self.calls += 1
        data = json.loads(user.split("Input:\n", 1)[1])
        c = self.calls

        def fake(lid, side, attrs):
            return {a: [f"FAKE-c{c}-{lid}-{side}-{a}-{i}" for i in range(len(v))]
                    for a, v in attrs.items()}

        sides = [s for s in ("KG1", "KG2") if s in data["S"]]
        if "Create ONE new" in system:      # all_llm prompt
            out = {s: {"attributes": fake("S", s, data["S"][s]["attributes"]),
                       "relations": data["S"][s]["relations"]} for s in sides}
        else:
            out = {} if "ALREADY been replaced" in system else {
                "S": {s: fake("S", s, data["S"][s]["attributes"]) for s in sides}}
            for lid, nb in data["neighbors"].items():
                out[lid] = {s: fake(lid, s, nb[s]["attributes"]) for s in ("KG1", "KG2") if s in nb}
        return LLMResponse(text=json.dumps(out), from_cache=False)


class MockBackend:
    """Offline backend: ``fn(system, user) -> str``."""

    def __init__(self, fn: Callable[[str, str], str]):
        self.fn = fn

    def __call__(self, system: str, user: str) -> LLMResponse:
        return LLMResponse(text=self.fn(system, user), from_cache=False)


# ----------------------------------------------------------------------
# Front-end used by the augmenter
# ----------------------------------------------------------------------
class CachedLLMClient:
    def __init__(self, backend, cache: ResponseCache, model_id: str, params: dict, seed: int):
        self.backend = backend
        self.cache = cache
        self.model_id = model_id
        self.params = dict(params)
        self.seed = int(seed)
        self.n_api_calls = 0
        self.n_disk_hits = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.n_truncated_outputs = 0   # stopReason == max_tokens (answer cut, JSON likely invalid)

    def generate(self, system: str, user: str) -> LLMResponse:
        key = cache_key(self.model_id, self.params, self.seed, system, user)
        cached = self.cache.get(key)
        if cached is not None:
            self.n_disk_hits += 1
            return LLMResponse(text=cached, from_cache=True)
        response = self.backend(system, user)
        self.n_api_calls += 1
        self.input_tokens += response.input_tokens
        self.output_tokens += response.output_tokens
        if response.stop_reason == "max_tokens":
            self.n_truncated_outputs += 1
        self.cache.put(key, model_id=self.model_id, seed=self.seed, params=self.params,
                       system=system, user=user, response=response)
        return response


def build_client(llm_cfg: dict, seed: int, mock_fn: Optional[Callable[[str, str], str]] = None
                 ) -> CachedLLMClient:
    """Create the client described by the ``augmentation.llm`` config block."""
    provider = llm_cfg.get("provider", "bedrock")
    model_id = llm_cfg["model_id"]
    if provider == "fake" or (provider == "replay" and llm_cfg.get("replay_fake", False)):
        # Placeholder answers live in their own key space, so they can never be
        # served to a real run (replay of a dry run needs replay_fake: true).
        model_id = f"fake::{model_id}"
    params = {
        "temperature": float(llm_cfg.get("temperature", 0.7)),
        "top_p": float(llm_cfg.get("top_p", 0.9)),
        "max_tokens": int(llm_cfg.get("max_tokens", 8192)),
    }
    cache = ResponseCache(llm_cfg["cache_path"])
    if provider == "bedrock":
        backend = BedrockBackend(
            model_id, llm_cfg.get("region", "us-east-1"), params,
            requests_per_second=float(llm_cfg.get("requests_per_second", 2.0)),
            max_retries=int(llm_cfg.get("max_retries", 8)),
        )
    elif provider == "replay":
        backend = ReplayBackend()
    elif provider == "fake":
        backend = FakeBackend()
    elif provider == "mock":
        if mock_fn is None:
            raise ValueError("provider=mock requires mock_fn")
        backend = MockBackend(mock_fn)
    else:
        raise ValueError(f"Unknown llm.provider '{provider}' (bedrock | replay | fake | mock)")
    return CachedLLMClient(backend, cache, model_id, params, seed)
