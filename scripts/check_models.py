"""Diagnose which free LLM endpoint actually answers from CI (no secret is printed).

The script repeats the real bot request against GitHub Models and, when that host is
unavailable, checks a few keyless community endpoints and the GitHub API itself for
comparison. Results are published as a pull-request comment.

Run:  python scripts/check_models.py
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402

from service_bot.ai import SYSTEM_PROMPT, knowledge_digest  # noqa: E402
from service_bot.knowledge import KnowledgeStore  # noqa: E402
from service_bot.llm import POLLINATIONS_BASE_URL, resolve_llm_config  # noqa: E402

GITHUB_MODELS_BASE = "https://models.github.ai/inference"
TOKEN_VARS = ("GH_MODELS_TOKEN", "GITHUB_TOKEN", "GH_TOKEN")
PREVIEW = 200
PROBE_QUESTIONS = "Как почистить КАН Ультра?"


def find_token() -> str:
    for name in TOKEN_VARS:
        value = (os.getenv(name) or "").strip()
        if value:
            print(f"token: present in {name} ({len(value)} chars), value never printed")
            return value
    print("token: none found in the environment")
    return ""


def report(response: httpx.Response, label: str) -> str:
    body = " ".join(response.text.split())
    print(
        f"{label}: HTTP {response.status_code} "
        f"content-type={response.headers.get('content-type', '-')} body={body[:PREVIEW]!r}"
    )
    return body


def probe(url: str, payload: dict, headers: dict, label: str) -> tuple[bool, str]:
    try:
        response = httpx.post(url, headers=headers, json=payload, timeout=40)
    except httpx.HTTPError as exc:
        print(f"{label}: ERROR {type(exc).__name__}")
        return False, ""
    report(response, label)
    if response.status_code != 200:
        return False, ""
    try:
        content = response.json()["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError):
        print(f"{label}: 200 без choices[0].message.content — ответ не от модели")
        return False, ""
    print(f"{label}: content={content[:160]!r}")
    return True, content


def environment_note(token: str) -> None:
    try:
        print(f"dns models.github.ai -> {socket.gethostbyname('models.github.ai')}")
    except OSError as exc:
        print(f"dns models.github.ai ERROR {type(exc).__name__}")
    try:
        base = httpx.get("https://api.github.com/rate_limit", timeout=25)
        print(f"github api: HTTP {base.status_code} (сеть GitHub в порядке)")
    except httpx.HTTPError as exc:
        print(f"github api: ERROR {type(exc).__name__}")
    try:
        root = httpx.get("https://models.github.ai/", timeout=25)
        report(root, "github models: GET /")
    except httpx.HTTPError as exc:
        print(f"github models: GET / ERROR {type(exc).__name__}")
    command = [
        "curl",
        "-sS",
        "-o",
        "/dev/stderr",
        "-w",
        "curl: HTTP %{http_code}\\n",
        f"{GITHUB_MODELS_BASE}/chat/completions",
        "-H",
        "Content-Type: application/json",
        "-H",
        f"Authorization: Bearer {token}",
        "-d",
        '{"messages":[{"role":"user","content":"ок"}],"model":"openai/gpt-4o-mini"}',
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=45, check=False)
        print(f"curl: {result.stdout.strip()[:80]!r} {' '.join(result.stderr.split())[:160]!r}")
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"curl: ERROR {type(exc).__name__}")


def probe_keyless() -> bool:
    """Best-effort keyless community endpoints; used only for the demo."""
    found = probe(
        "https://api.llm7.io/v1/chat/completions",
        {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "ответь: ок"}]},
        {"Content-Type": "application/json"},
        "llm7.io",
    )[0]
    try:
        status = httpx.get(
            "https://duckduckgo.com/duckchat/v1/status",
            headers={"x-vqd-accept": "1", "User-Agent": "Mozilla/5.0"},
            timeout=25,
        )
        print(f"duckduckgo: status HTTP {status.status_code}")
        vqd = status.headers.get("x-vqd-4")
        if vqd:
            chat = httpx.post(
                "https://duckduckgo.com/duckchat/v1/chat",
                headers={
                    "x-vqd-4": vqd,
                    "Content-Type": "application/json",
                    "User-Agent": "Mozilla/5.0",
                },
                json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "ок"}]},
                timeout=40,
            )
            print(f"duckduckgo: chat HTTP {chat.status_code} {chat.text[:160]!r}")
            found = found or chat.status_code == 200
    except httpx.HTTPError as exc:
        print(f"duckduckgo: ERROR {type(exc).__name__}")
    pollinations = probe(
        POLLINATIONS_BASE_URL,
        {"model": "openai", "messages": [{"role": "user", "content": "ответь: ок"}]},
        {"Content-Type": "application/json"},
        "pollinations: minimal",
    )[0]
    # Extra parameters are what the free tier rejected earlier; the bot now omits them.
    probe(
        POLLINATIONS_BASE_URL,
        {
            "model": "openai",
            "messages": [{"role": "user", "content": "ответь: ок"}],
            "temperature": 0.0,
            "max_tokens": 300,
            "response_format": {"type": "json_object"},
        },
        {"Content-Type": "application/json"},
        "pollinations: full params",
    )
    return found or pollinations


def main() -> int:
    token = find_token()
    config = resolve_llm_config(os.environ)
    model = config.model if config else "openai/gpt-4o-mini"
    digest = knowledge_digest(KnowledgeStore.load())
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(digest=digest)},
        {
            "role": "user",
            "content": f"# Вопрос клиента\n{PROBE_QUESTIONS}\n\n# Контекст диалога\nнет",
        },
    ]
    exact = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 300,
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    environment_note(token)
    ok, _ = probe(f"{GITHUB_MODELS_BASE}/chat/completions", exact, headers, "github: exact")
    if not ok:
        short = {"model": model, "messages": [{"role": "user", "content": "ок"}], "max_tokens": 5}
        ok, _ = probe(f"{GITHUB_MODELS_BASE}/chat/completions", short, headers, "github: minimal")
    if not ok:
        ok = probe_keyless()
    print("Итог: внешняя модель доступна" if ok else "Итог: внешние модели недоступны")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
