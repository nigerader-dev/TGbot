import pytest

from service_bot.ai import AIAssistant
from service_bot.engine import AnswerEngine
from service_bot.knowledge import KnowledgeStore

# Ambient credentials (GH_TOKEN, GITHUB_TOKEN, ...) must never reach tests:
# an accidental real request would be slow, flaky and unrelated to the assertions.
LLM_ENV_VARS = (
    "AI_ENABLED",
    "LLM_PROVIDER",
    "LLM_API_KEY",
    "LLM_BASE_URL",
    "LLM_MODEL",
    "LLM_TIMEOUT",
    "LLM_RETRIES",
    "LLM_COOLDOWN_SECONDS",
    "OPENAI_API_KEY",
    "GH_MODELS_TOKEN",
    "GITHUB_TOKEN",
    "GH_TOKEN",
    "OPENROUTER_API_KEY",
    "GROQ_API_KEY",
)


@pytest.fixture(autouse=True)
def _isolated_llm_environment(monkeypatch):
    for name in LLM_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def knowledge():
    return KnowledgeStore.load()


@pytest.fixture
def engine(knowledge):
    return AnswerEngine(knowledge)


@pytest.fixture
def assistant(engine):
    """Rules-only assistant: the same object Telegram and the demo use."""
    return AIAssistant(engine)
