"""A same-origin live demo using the exact same policy as the Telegram adapter."""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Annotated
from uuid import UUID

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, StringConstraints

from .ai import AIAssistant
from .engine import AnswerEngine
from .knowledge import DEFAULT_KNOWLEDGE_PATH, KnowledgeStore
from .llm import LLMRouter, resolve_llm_configs
from .telegram import TelegramBot, TelegramStatus

STATIC = Path(__file__).resolve().parent / "static"


class ChatInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1500)]
    session_id: UUID


class SessionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    session_id: UUID


def build_assistant(
    knowledge: KnowledgeStore,
    *,
    env: dict | None = None,
    client: httpx.AsyncClient | None = None,
) -> AIAssistant:
    """Wire the optional LLM chain. Without a provider the rule engine answers."""
    configs = resolve_llm_configs(env)
    routers = [LLMRouter(config, client=client) for config in configs]
    return AIAssistant(
        AnswerEngine(knowledge),
        router=routers[0] if routers else None,
        config=configs[0] if configs else None,
        fallback_routers=routers[1:],
        fallback_configs=configs[1:],
    )


def create_app(
    *,
    knowledge: KnowledgeStore | None = None,
    telegram_token: str | None = None,
    assistant: AIAssistant | None = None,
) -> FastAPI:
    if knowledge is None:
        path = Path(os.getenv("KNOWLEDGE_BASE_PATH") or DEFAULT_KNOWLEDGE_PATH)
        knowledge = KnowledgeStore.load(path)
    if assistant is None:
        assistant = build_assistant(knowledge)
    engine = assistant.engine
    token = os.getenv("TELEGRAM_BOT_TOKEN", "") if telegram_token is None else telegram_token
    bot = TelegramBot(token, assistant) if token.strip() else None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        task = asyncio.create_task(bot.run()) if bot else None
        yield
        if task:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        await assistant.aclose()

    app = FastAPI(title="Сервисный помощник", docs_url=None, redoc_url=None, lifespan=lifespan)
    app.state.engine = engine
    app.state.assistant = assistant
    app.state.telegram = bot

    @app.middleware("http")
    async def small_requests_and_safe_headers(request: Request, call_next):
        length = request.headers.get("content-length", "0")
        if not length.isdigit() or int(length) > 8192:
            return JSONResponse({"detail": "Request is too large"}, status_code=413)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        # No frame-ancestors / X-Frame-Options restriction: Arena's preview is embedded.
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "img-src 'self' data:; connect-src 'self'; object-src 'none'; "
            "base-uri 'none'; form-action 'self'"
        )
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/")
    async def home():
        return FileResponse(STATIC / "index.html")

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "knowledge_revision": knowledge.document.revision,
            "ai_enabled": assistant.enabled,
        }

    @app.get("/api/knowledge")
    async def get_knowledge():
        return knowledge.public_view()

    @app.get("/api/status")
    async def get_status():
        return {
            "mode": "telegram_and_demo" if bot else "demo",
            "telegram": (bot.status if bot else TelegramStatus()).public_view(),
            "knowledge_revision": knowledge.document.revision,
            "entry_count": len(knowledge.entries),
            "policy": "verbatim_knowledge_only",
            "ai": assistant.status(),
        }

    @app.post("/api/chat")
    async def chat(data: ChatInput):
        return (await assistant.respond(data.text, f"web:{data.session_id}")).to_dict()

    @app.post("/api/reset")
    async def reset(data: SessionInput):
        return engine.reset(f"web:{data.session_id}").to_dict()

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app
