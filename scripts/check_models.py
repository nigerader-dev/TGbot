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


def probe_keyless(digest: str) -> bool:
    """Find out what the keyless endpoint accepts; used only for the demo tier."""
    headers = {"Content-Type": "application/json"}
    short = {"role": "user", "content": "ответь одним словом: ок"}
    system = {"role": "system", "content": "Ты — маршрутизатор базы знаний."}
    minimal = probe(
        POLLINATIONS_BASE_URL,
        {"model": "openai", "messages": [short]},
        headers,
        "pollinations: minimal",
    )[0]
    with_accept = probe(
        POLLINATIONS_BASE_URL,
        {"model": "openai", "messages": [short]},
        {"Content-Type": "application/json", "Accept": "application/json"},
        "pollinations: +Accept header",
    )[0]
    prompt_variants = {
        "system short + user": [system, short],
        "system digest + user": [
            {"role": "system", "content": SYSTEM_PROMPT.format(digest=digest)},
            {
                "role": "user",
                "content": "# Вопрос клиента\nКак почистить КАН Ультра?\n\n# Контекст диалога\nнет",
            },
        ],
        "digest inside user": [
            {
                "role": "user",
                "content": (
                    "База знаний:\n" + digest + "\n\nВопрос клиента: Как почистить КАН Ультра?\n"
                    "Ответь JSON-объектом."
                ),
            }
        ],
    }
    ok = minimal or with_accept
    for label, messages in prompt_variants.items():
        result = probe(
            POLLINATIONS_BASE_URL,
            {"model": "openai", "messages": messages},
            headers,
            f"pollinations: {label}",
        )[0]
        ok = ok or result
    try:
        models = httpx.get("https://text.pollinations.ai/models", timeout=25)
        print(f"pollinations: GET /models HTTP {models.status_code} {models.text[:200]!r}")
    except httpx.HTTPError as exc:
        print(f"pollinations: GET /models ERROR {type(exc).__name__}")
    try:
        available = httpx.get("https://api.llm7.io/v1/models", timeout=25)
        models = available.json().get("data", []) if available.status_code == 200 else []
        names = [item.get("id") for item in models][:12]
        print(f"llm7.io: GET /v1/models HTTP {available.status_code} {names}")
        for name in names[:3]:
            if name:
                probe(
                    "https://api.llm7.io/v1/chat/completions",
                    {"model": name, "messages": [{"role": "user", "content": "ответь: ок"}]},
                    headers,
                    f"llm7.io {name}",
                )
    except (httpx.HTTPError, ValueError) as exc:
        print(f"llm7.io: GET /v1/models ERROR {type(exc).__name__}")
    return ok


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
        ok = probe_keyless(digest)
    print("Итог: внешняя модель доступна" if ok else "Итог: внешние модели недоступны")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
