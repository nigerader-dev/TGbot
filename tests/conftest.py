import pytest

from service_bot.engine import AnswerEngine
from service_bot.knowledge import KnowledgeStore


@pytest.fixture
def knowledge():
    return KnowledgeStore.load()


@pytest.fixture
def engine(knowledge):
    return AnswerEngine(knowledge)
