"""Diagnose GitHub Models access from CI without printing any secret.

Prints the HTTP status and a short body for a few model names and both known base
URLs, so a failing AI layer can be debugged from a pull-request comment.

Run:  python scripts/check_models.py
"""

from __future__ import annotations

import os
import sys

import httpx

BASE_URLS = ("https://models.github.ai/inference", "https://models.inference.ai.azure.com")
MODELS = ("openai/gpt-4o-mini", "openai/gpt-4.1-mini", "openai/gpt-4o")
TOKEN_VARS = ("GH_MODELS_TOKEN", "GITHUB_TOKEN", "GH_TOKEN")


def find_token() -> str:
    for name in TOKEN_VARS:
        value = (os.getenv(name) or "").strip()
        if value:
            print(f"token: present in {name} ({len(value)} chars), value never printed")
            return value
    print("token: missing (set GH_MODELS_TOKEN or LLM_API_KEY)")
    return ""


def main() -> int:
    token = find_token()
    if not token:
        return 1
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {
        "model": MODELS[0],
        "messages": [{"role": "user", "content": "ответь одним словом: ок"}],
        "max_tokens": 5,
    }
    ok = False
    for base_url in BASE_URLS:
        for model in MODELS:
            body = {**payload, "model": model}
            try:
                response = httpx.post(
                    f"{base_url}/chat/completions", headers=headers, json=body, timeout=20
                )
            except httpx.HTTPError as exc:
                print(f"{base_url} {model}: ERROR {type(exc).__name__}")
                continue
            preview = " ".join(response.text.split())[:180]
            print(f"{base_url} {model}: HTTP {response.status_code} {preview}")
            ok = ok or response.status_code == 200
    print("GitHub Models: доступен" if ok else "GitHub Models: недоступен")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
