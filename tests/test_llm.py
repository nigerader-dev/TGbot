import asyncio
import json

import httpx
import pytest

from service_bot.llm import (
    LLMConfig,
    LLMRouter,
    parse_verdict,
    resolve_llm_config,
)

FAKE_KEY = "sk-test-key-must-never-be-published"


def config(**overrides) -> LLMConfig:
    data = {
        "provider": "github",
        "base_url": "https://models.github.ai/inference",
        "model": "openai/gpt-4o-mini",
        "api_key": FAKE_KEY,
        "timeout": 5.0,
        "retries": 1,
    }
    data.update(overrides)
    return LLMConfig(**data)


def completion(content: str, status: int = 200) -> httpx.Response:
    if status != 200:
        return httpx.Response(status, json={"error": {"message": "boom"}})
    return httpx.Response(
        200, json={"choices": [{"message": {"role": "assistant", "content": content}}]}
    )


# -- configuration -----------------------------------------------------------


def test_auto_prefers_an_explicit_key():
    resolved = resolve_llm_config({"LLM_API_KEY": FAKE_KEY})
    assert resolved is not None
    assert (resolved.provider, resolved.model) == ("openai", "gpt-4o-mini")
    assert resolved.public_view()["key_configured"] is True
    assert FAKE_KEY not in json.dumps(resolved.public_view())


def test_auto_falls_back_to_github_models_with_a_workflow_token():
    resolved = resolve_llm_config({"GH_MODELS_TOKEN": "ghs_example"})
    assert resolved is not None
    assert resolved.provider == "github"
    assert resolved.base_url == "https://models.github.ai/inference"
    assert resolved.model.startswith("openai/")


def test_auto_without_credentials_uses_the_keyless_demo_endpoint():
    resolved = resolve_llm_config({})
    assert resolved is not None
    assert resolved.provider == "llm7"
    assert resolved.base_url == "https://api.llm7.io/v1"
    assert resolved.model == "GLM-5.3-Flash"
    assert resolved.api_key == ""
    assert resolved.needs_key is False


def test_pollinations_stays_available_as_a_keyless_fallback():
    resolved = resolve_llm_config({"LLM_PROVIDER": "pollinations"})
    assert resolved is not None
    assert resolved.api_key == ""
    assert resolved.minimal_payload is True  # some params trigger a paid path there
    assert resolved.extra_body == {}


@pytest.mark.parametrize("value", ["0", "off", "false", "disabled"])
def test_ai_can_be_switched_off(value):
    assert resolve_llm_config({"AI_ENABLED": value, "LLM_API_KEY": "k"}) is None
    assert resolve_llm_config({"LLM_PROVIDER": value, "LLM_API_KEY": "k"}) is None


@pytest.mark.parametrize(
    "environment",
    [
        {"LLM_PROVIDER": "openai"},  # key required
        {"LLM_PROVIDER": "custom"},  # base url required
        {"LLM_PROVIDER": "unknown-provider"},
    ],
)
def test_incomplete_configuration_disables_the_ai_layer(environment):
    assert resolve_llm_config(environment) is None


def test_explicit_provider_overrides_and_limits_are_clamped():
    resolved = resolve_llm_config(
        {
            "LLM_PROVIDER": "ollama",
            "LLM_MODEL": "qwen2.5:3b",
            "LLM_TIMEOUT": "9999",
            "LLM_COOLDOWN_SECONDS": "1",
            "LLM_RETRIES": "9",
        }
    )
    assert resolved is not None
    assert resolved.model == "qwen2.5:3b"
    assert resolved.timeout == 120.0
    assert resolved.cooldown == 5.0
    assert resolved.retries == 3


# -- verdict parsing ---------------------------------------------------------


def test_parse_verdict_accepts_plain_json():
    verdict = parse_verdict('{"action": "answer", "entry_id": "kit_frequency", "confidence": 0.8}')
    assert verdict is not None
    assert (verdict.action, verdict.entry_id, verdict.confidence) == (
        "answer",
        "kit_frequency",
        0.8,
    )


def test_parse_verdict_tolerates_fences_and_prose():
    content = 'Конечно!\n```json\n{"action":"no_answer","reason":"нет темы"}\n```\nГотово.'
    verdict = parse_verdict(content)
    assert verdict is not None
    assert verdict.action == "no_answer"
    assert verdict.reason == "нет темы"


@pytest.mark.parametrize(
    "content",
    [
        "я не понял",
        '{"action": "invent"}',
        "",
        None,
        42,
        '{"action": "answer"}',
        '{"action":"answer","entry_id":"kan_ultra_maintenance"}',
    ],
)
def test_parse_verdict_rejects_anything_unexpected(content):
    verdict = parse_verdict(content)
    assert verdict is None or verdict.action == "answer"


@pytest.mark.parametrize(
    "alias,expected",
    [("no_match", "no_answer"), ("missing", "no_answer"), ("ask", "clarify"), ("select", "answer")],
)
def test_parse_verdict_normalizes_aliases(alias, expected):
    verdict = parse_verdict({"action": alias, "entry_id": "kit_frequency"})
    assert verdict is not None
    assert verdict.action == expected


def test_parse_verdict_clamps_confidence_and_trims_reason():
    content = {"action": "answer", "entry_id": "x", "confidence": 7, "reason": "  a\nb  "}
    verdict = parse_verdict(content)
    assert verdict is not None
    assert verdict.confidence == 1.0
    assert verdict.reason == "a b"


# -- transport ---------------------------------------------------------------


def route(router, system="system", user="user"):
    return asyncio.run(router.route(system=system, user=user))


def test_router_sends_the_expected_payload_and_reads_the_verdict():
    requests = []

    def handler(request):
        requests.append(
            {
                "url": str(request.url),
                "auth": request.headers.get("authorization"),
                "body": json.loads(request.content),
            }
        )
        return completion('{"action":"answer","entry_id":"kan_ultra_maintenance"}')

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    router = LLMRouter(config(), client=client)
    verdict = route(router, system="СИСТЕМА", user="ВОПРОС")
    assert verdict is not None
    assert verdict.action == "answer"
    assert verdict.entry_id == "kan_ultra_maintenance"
    assert verdict.latency_ms >= 0
    assert requests[0]["url"] == "https://models.github.ai/inference/chat/completions"
    assert requests[0]["auth"] == f"Bearer {FAKE_KEY}"
    assert requests[0]["body"]["model"] == "openai/gpt-4o-mini"
    assert requests[0]["body"]["response_format"] == {"type": "json_object"}
    assert requests[0]["body"]["messages"][0]["content"] == "СИСТЕМА"
    assert requests[0]["body"]["messages"][1]["content"] == "ВОПРОС"
    assert router.state == "ready"
    assert router.answered == 1
    public = json.dumps(router.public_view(), ensure_ascii=False)
    assert FAKE_KEY not in public
    asyncio.run(router.aclose())


def test_router_retries_server_errors_then_succeeds():
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return completion("", status=503)
        return completion('{"action":"clarify"}')

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    router = LLMRouter(config(retries=1), client=client)
    verdict = route(router)
    assert verdict is not None
    assert verdict.action == "clarify"
    assert len(calls) == 2
    assert router.clarified == 1
    asyncio.run(router.aclose())


def test_router_returns_none_and_counts_a_fallback_on_persistent_failure():
    def handler(request):
        return completion("", status=500)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    router = LLMRouter(config(retries=0), client=client)
    assert route(router) is None
    assert router.last_error == "http_500"
    assert router.fallbacks == 1
    asyncio.run(router.aclose())


def test_breaker_counts_failures_before_tripping():
    calls = []

    def handler(request):
        calls.append(request)
        return completion("", status=500)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    router = LLMRouter(config(retries=0), client=client)
    assert route(router) is None
    assert router.state == "ready"
    assert len(calls) == 1
    assert router.last_error == "http_500"
    asyncio.run(router.aclose())


def test_breaker_cools_down_after_repeated_failures():
    clock = {"now": 0.0}
    calls = []

    def handler(request):
        calls.append(request)
        return completion("", status=500)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    router = LLMRouter(config(retries=0, cooldown=30.0), client=client, clock=lambda: clock["now"])
    for _ in range(LLMRouter.FAILURE_THRESHOLD):
        assert route(router) is None
    assert router.state == "cooling_down"
    before = len(calls)
    assert route(router) is None
    assert len(calls) == before  # no request left the process
    clock["now"] = 60.0
    assert router.state == "ready"
    asyncio.run(router.aclose())


def test_network_errors_are_sanitized_and_never_leak_the_key():
    def handler(request):
        raise httpx.ConnectError(f"Cannot connect to {request.url}", request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    router = LLMRouter(config(retries=0), client=client)
    assert route(router) is None
    assert router.last_error == "network"
    assert FAKE_KEY not in json.dumps(router.public_view(), ensure_ascii=False)
    asyncio.run(router.aclose())


@pytest.mark.parametrize(
    "response,expected",
    [
        (httpx.Response(200, text="not-json"), "bad_response"),
        (httpx.Response(200, json={"choices": []}), "bad_response"),
        (completion("не JSON вовсе"), "bad_verdict"),
    ],
)
def test_broken_model_answers_are_reported_as_kinds(response, expected):
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: response))
    router = LLMRouter(config(retries=0), client=client)
    assert route(router) is None
    assert router.last_error == expected
    asyncio.run(router.aclose())


def test_router_is_idle_when_the_ai_layer_is_off():
    router = LLMRouter(config(provider="openai", api_key=""))
    assert route(router) is None
