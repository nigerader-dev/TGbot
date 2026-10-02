"""Local stand-in for an LLM, used only for offline previews (no network available).

It implements the same OpenAI-compatible contract as a real provider and answers with
the same JSON verdicts a well-behaved small model would produce. It exists so the demo
can show the full pipeline (model decides -> answer copied from the knowledge base)
inside sandboxes without internet access. The real bot uses a real model; see README.

Run:  python scripts/preview_llm_stub.py --port 8899
"""

from __future__ import annotations

import argparse
import json

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict

app = FastAPI(title="offline preview model", docs_url=None, redoc_url=None)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str = "preview-stub"
    messages: list[dict]


def verdict(question: str) -> dict:
    text = question.casefold()
    if "тверь" in text or "войну" in text or "телефон" in text:
        return {"action": "no_answer", "reason": "нет данных в базе"}
    if "переполн" in text or "затопл" in text or "перелив" in text:
        return {"action": "answer", "entry_id": "station_overflow", "confidence": 0.93}
    if "кит" in text and ("част" in text or "год" in text or "період" in text or "период" in text):
        return {"action": "answer", "entry_id": "kit_frequency", "confidence": 0.91}
    care = ("инструкц", "обслуж", "чист", "помы", "уход", "видео", "видос", "промы")
    if "кан" in text and any(word in text for word in care):
        return {"action": "answer", "entry_id": "kan_ultra_maintenance", "confidence": 0.95}
    if any(word in text for word in care):
        return {"action": "clarify", "reason": "модель станции не названа"}
    return {"action": "no_answer", "reason": "нет данных в базе"}


def extract_question(user_message: str) -> str:
    """The assistant sends "question + dialogue context"; keep the question only."""
    body = user_message.split("# Контекст диалога")[0]
    lines = [line.strip() for line in body.splitlines()[1:] if line.strip()]
    return lines[-1] if lines else ""


@app.post("/v1/chat/completions")
async def completions(payload: ChatRequest):
    user = payload.messages[-1]["content"] if payload.messages else ""
    content = json.dumps(verdict(extract_question(user)), ensure_ascii=False)
    return {
        "id": "preview-stub",
        "object": "chat.completion",
        "model": payload.model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Офлайн-заглушка ИИ-модели для превью")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8899)
    args = parser.parse_args()
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, access_log=False)


if __name__ == "__main__":
    main()
