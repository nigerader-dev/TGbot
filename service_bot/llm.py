"""OpenAI-compatible LLM transport: provider resolution, strict JSON verdicts, breaker.

The transport never decides what the client is told. It returns a structured verdict
(which knowledge entry is relevant, or "no answer"), and the caller validates it
against the knowledge base. The API key never leaves the process and is never logged.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import httpx

OPENAI_BASE_URL = "https://api.openai.com/v1"
GITHUB_MODELS_BASE_URL = "https://models.github.ai/inference"
POLLINATIONS_BASE_URL = "https://text.pollinations.ai/openai"
OLLAMA_BASE_URL = "http://127.0.0.1:11434/v1"

ACTIONS = {"answer", "clarify", "no_answer"}
ACTION_ALIASES = {
    "answer": "answer",
    "select": "answer",
    "clarify": "clarify",
    "ask": "clarify",
    "no_answer": "no_answer",
    "noanswer": "no_answer",
    "none": "no_answer",
    "missing": "no_answer",
    "no_match": "no_answer",
}


@dataclass(frozen=True)
class ProviderPreset:
    base_url: str
    model: str
    key_env: tuple[str, ...]
    needs_key: bool
    json_mode: bool = True
    extra_body: Mapping[str, Any] = field(default_factory=dict)


PRESETS: dict[str, ProviderPreset] = {
    "openai": ProviderPreset(
        OPENAI_BASE_URL, "gpt-4o-mini", ("LLM_API_KEY", "OPENAI_API_KEY"), True
    ),
    "github": ProviderPreset(
        GITHUB_MODELS_BASE_URL,
        "openai/gpt-4o-mini",
        ("GH_MODELS_TOKEN", "GITHUB_TOKEN", "GH_TOKEN"),
        True,
    ),
    "openrouter": ProviderPreset(
        "https://openrouter.ai/api/v1",
        "meta-llama/llama-3.1-8b-instruct",
        ("LLM_API_KEY", "OPENROUTER_API_KEY"),
        True,
    ),
    "groq": ProviderPreset(
        "https://api.groq.com/openai/v1",
        "llama-3.1-8b-instant",
        ("LLM_API_KEY", "GROQ_API_KEY"),
        True,
    ),
    "deepseek": ProviderPreset(
        "https://api.deepseek.com/v1", "deepseek-chat", ("LLM_API_KEY", "DEEPSEEK_API_KEY"), True
    ),
    "mistral": ProviderPreset(
        "https://api.mistral.ai/v1",
        "mistral-small-latest",
        ("LLM_API_KEY", "MISTRAL_API_KEY"),
        True,
    ),
    "gemini": ProviderPreset(
        "https://generativelanguage.googleapis.com/v1beta/openai",
        "gemini-2.0-flash",
        ("LLM_API_KEY", "GEMINI_API_KEY"),
        True,
    ),
    "ollama": ProviderPreset(OLLAMA_BASE_URL, "llama3.2", (), False, json_mode=False),
    "pollinations": ProviderPreset(
        POLLINATIONS_BASE_URL,
        "openai",
        (),
        False,
        json_mode=False,
        extra_body={"referrer": "github.com/nigerader-dev/TGbot"},
    ),
    "custom": ProviderPreset("", "", ("LLM_API_KEY",), True),
}
DISABLED_VALUES = {"0", "false", "no", "off", "none", "disabled", ""}


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    base_url: str
    model: str
    api_key: str
    timeout: float = 20.0
    retries: int = 1
    cooldown: float = 300.0
    max_tokens: int = 300
    temperature: float = 0.0
    json_mode: bool = True
    extra_body: Mapping[str, Any] = field(default_factory=dict)

    def public_view(self) -> dict:
        """Connection metadata only: the key is never part of any public payload."""
        return {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "key_configured": bool(self.api_key),
            "timeout_seconds": self.timeout,
        }


def _first_env(env: Mapping[str, str], names: tuple[str, ...]) -> str:
    for name in names:
        value = (env.get(name) or "").strip()
        if value:
            return value
    return ""


def resolve_llm_config(env: Mapping[str, str] | None = None) -> LLMConfig | None:
    """Select a provider from the environment. Returns None when the AI layer is off.

    Order for ``LLM_PROVIDER=auto`` (default): explicit key -> GitHub token
    (GitHub Models, free with a workflow ``models: read`` permission) -> keyless
    community endpoint used only as a best-effort demo.
    """
    environment = os.environ if env is None else env
    provider = (environment.get("LLM_PROVIDER") or "auto").strip().lower()
    if provider in DISABLED_VALUES or (
        environment.get("AI_ENABLED", "").strip().lower() in DISABLED_VALUES - {""}
    ):
        return None
    if provider == "auto":
        if _first_env(environment, ("LLM_API_KEY", "OPENAI_API_KEY")):
            provider = "openai"
        elif _first_env(environment, ("GH_MODELS_TOKEN", "GITHUB_TOKEN", "GH_TOKEN")):
            provider = "github"
        else:
            provider = "pollinations"
    preset = PRESETS.get(provider)
    if preset is None:
        return None
    base_url = (environment.get("LLM_BASE_URL") or preset.base_url).strip().rstrip("/")
    model = (environment.get("LLM_MODEL") or preset.model).strip()
    if not base_url or not model:
        return None
    api_key = _first_env(environment, preset.key_env) or (
        (environment.get("LLM_API_KEY") or "").strip() if preset.needs_key else ""
    )
    if preset.needs_key and not api_key:
        return None
    timeout = _positive_float(environment.get("LLM_TIMEOUT"), 20.0, 1.0, 120.0)
    cooldown = _positive_float(environment.get("LLM_COOLDOWN_SECONDS"), 300.0, 5.0, 86400.0)
    retries = int(_positive_float(environment.get("LLM_RETRIES"), 1.0, 0.0, 3.0))
    return LLMConfig(
        provider=provider,
        base_url=base_url,
        model=model,
        api_key=api_key,
        timeout=timeout,
        retries=retries,
        cooldown=cooldown,
        json_mode=preset.json_mode,
        extra_body=dict(preset.extra_body),
    )


def _positive_float(raw: str | None, default: float, low: float, high: float) -> float:
    try:
        value = float(raw) if raw not in (None, "") else default
    except (TypeError, ValueError):
        return default
    return min(max(value, low), high)


def _first_json_object(text: str) -> dict | None:
    """Return the first balanced JSON object, tolerating code fences and prose."""
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : index + 1]
                    try:
                        parsed = json.loads(candidate)
                    except ValueError:
                        break
                    if isinstance(parsed, dict):
                        return parsed
                    break
        start = text.find("{", start + 1)
    return None


@dataclass(frozen=True)
class RouteVerdict:
    action: str
    entry_id: str | None = None
    confidence: float | None = None
    reason: str | None = None
    latency_ms: int = 0


def parse_verdict(content: Any) -> RouteVerdict | None:
    """Parse a model answer into a verdict. Anything unexpected returns None."""
    if isinstance(content, str):
        payload = _first_json_object(content)
    elif isinstance(content, dict):
        payload = content
    else:
        payload = None
    if not isinstance(payload, dict):
        return None
    action = str(payload.get("action", "")).strip().lower()
    action = ACTION_ALIASES.get(action, "")
    if action not in ACTIONS:
        return None
    entry_id = payload.get("entry_id")
    entry_id = entry_id.strip() if isinstance(entry_id, str) and entry_id.strip() else None
    confidence = payload.get("confidence")
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = None
    if confidence is not None:
        confidence = min(max(confidence, 0.0), 1.0)
    reason = payload.get("reason")
    reason = " ".join(str(reason).split())[:160] if reason else None
    return RouteVerdict(action=action, entry_id=entry_id, confidence=confidence, reason=reason)


class LLMRouter:
    """Transport only: builds requests, parses verdicts, trips a breaker on failures."""

    FAILURE_THRESHOLD = 3

    def __init__(
        self,
        config: LLMConfig,
        *,
        client: httpx.AsyncClient | None = None,
        clock=time.monotonic,
    ):
        self.config = config
        self._client = client
        self._clock = clock
        self._failures = 0
        self._disabled_until = 0.0
        self.last_error: str | None = None
        self.last_error_detail: str | None = None
        self.calls = 0
        self.answered = 0
        self.clarified = 0
        self.no_answer = 0
        self.fallbacks = 0

    def public_view(self) -> dict:
        return {
            **self.config.public_view(),
            "state": self.state,
            "last_error": self.last_error,
            "last_error_detail": self.last_error_detail,
            "calls": self.calls,
            "verdicts": {
                "answer": self.answered,
                "clarify": self.clarified,
                "no_answer": self.no_answer,
            },
            "fallbacks": self.fallbacks,
        }

    @property
    def state(self) -> str:
        if self._disabled_until and self._clock() < self._disabled_until:
            return "cooling_down"
        return "ready"

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _payload(self, system: str, user: str) -> dict:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        if self.config.json_mode:
            payload["response_format"] = {"type": "json_object"}
        payload.update(self.config.extra_body)
        return payload

    def _sanitize(self, text: str) -> str:
        """Short body preview for diagnostics; the key can never appear in it."""
        clean = " ".join(text.split())
        if self.config.api_key:
            clean = clean.replace(self.config.api_key, "<hidden>")
        return clean[:200]

    def _record_failure(self, kind: str, detail: str | None = None) -> None:
        self._failures += 1
        self.last_error = kind
        self.last_error_detail = self._sanitize(detail) if detail else None
        if self._failures >= self.FAILURE_THRESHOLD:
            self._disabled_until = self._clock() + self.config.cooldown
            self._failures = 0

    def _record_success(self, verdict: RouteVerdict) -> None:
        self._failures = 0
        self._disabled_until = 0.0
        self.last_error = None
        self.last_error_detail = None
        if verdict.action == "answer":
            self.answered += 1
        elif verdict.action == "clarify":
            self.clarified += 1
        else:
            self.no_answer += 1

    async def route(self, *, system: str, user: str) -> RouteVerdict | None:
        """Ask the model which entry is relevant. None means "use the rule engine"."""
        if not self.config.api_key and self.config.provider not in {"pollinations", "ollama"}:
            return None
        if self._disabled_until and self._clock() < self._disabled_until:
            self.fallbacks += 1
            return None
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(self.config.timeout, connect=10))
        started = self._clock()
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        for attempt in range(self.config.retries + 1):
            self.calls += 1
            try:
                response = await self._client.post(
                    f"{self.config.base_url}/chat/completions",
                    json=self._payload(system, user),
                    headers=headers,
                )
            except httpx.HTTPError:
                # Never surface exception strings: the request URL may reveal a token.
                self._record_failure("network")
                if attempt == self.config.retries:
                    break
                continue
            if response.status_code >= 400:
                kind = f"http_{response.status_code}"
                self._record_failure(kind, response.text)
                if response.status_code == 429 or response.status_code >= 500:
                    if attempt < self.config.retries:
                        continue
                break
            try:
                data = response.json()
                content = data["choices"][0]["message"]["content"]
            except (ValueError, KeyError, IndexError, TypeError):
                self._record_failure("bad_response", response.text)
                break
            verdict = parse_verdict(content)
            if verdict is None:
                self._record_failure("bad_verdict", str(content))
                break
            self._record_success(verdict)
            return RouteVerdict(
                action=verdict.action,
                entry_id=verdict.entry_id,
                confidence=verdict.confidence,
                reason=verdict.reason,
                latency_ms=int((self._clock() - started) * 1000),
            )
        self.fallbacks += 1
        return None
