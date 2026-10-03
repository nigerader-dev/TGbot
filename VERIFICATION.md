# Результаты проверки — 2026-10-02

Проверка выполнена на ветке `arena/01a0fe4e-tgbot` (PR #2), окружение: локальный
sandbox + настоящие GitHub Actions репозитория `nigerader-dev/TGbot`.

## Что проверено статически

- `python -m service_bot validate-kb`: база корректна — **3 записи, 2 модели,
  версия `prototype-2`**.
- `ruff check .` — без замечаний; `ruff format --check .` — 24 файла уже
  отформатированы.
- `pytest --cov=service_bot --cov-report=term-missing --cov-fail-under=90` —
  **274 passed**, покрытие **96%**.
- `node --check service_bot/static/app.js` — синтаксис JavaScript корректен.
- Тесты ИИ-слоя (`tests/test_ai.py`, `tests/test_llm.py`) используют
  `httpx.MockTransport`: реальных сетевых вызовов и ключей в тестах нет.
- Проверены все ключевые провайдеры: для каждого пресета (`openai`, `openrouter`,
  `groq`, `deepseek`, `mistral`, `gemini`) тест подтверждает, что секрет
  `LLM_API_KEY` подключает провайдера без правок кода, а ключ не попадает в
  публичные данные (`public_view`). Отдельно проверено, что режим `auto` узнаёт
  ключ `GEMINI_API_KEY`/`GROQ_API_KEY` без переменной `LLM_PROVIDER`, ключ с префиксом
  (`AIza…`, `gsk_…`, `sk-or-…`) и что JSON-режим автоматически снимается, если
  провайдер его не поддерживает.
- Модели у провайдеров сверены с официальными страницами (октябрь 2026):
  OpenAI `gpt-6-luna`, Google `gemini-3.5-flash-lite`, Groq `openai/gpt-oss-20b`
  (прежние `gpt-4o-mini`, `gemini-2.0-flash`, `llama-3.1-8b-instant` устарели).

## Веб-демо: полный путь «модель → база знаний»

Локальный запуск: `serve --host 0.0.0.0 --port 8000`, модель — офлайн-заглушка
`scripts/preview_llm_stub.py` (реализует тот же OpenAI-совместимый контракт, что и
настоящий провайдер). Проверено через реальный HTTP API `/api/chat`:

| Вопрос | Статус | Запись | Кто решил |
| --- | --- | --- | --- |
| «Как почистить КАН Ультра?» | answer | `kan_ultra_maintenance` | `layer=llm`, 6 мс |
| «как часто обслуживать КИТ» | answer | `kit_frequency` | `layer=llm`, 2 мс |
| «что делать если станция переполнена» | answer | `station_overflow` | `layer=llm`, 2 мс |
| «как самому обслужить станцию» | clarify | — | `layer=llm` (уточнение модели) |
| «как почистить станцию Тверь?» | missing | — | `layer=rules` (ИИ не спрашивали) |
| «кто написал войну и мир» | missing | — | `layer=rules` (тема вне базы) |

Тексты ответов совпадают с `answer` соответствующих записей базы посимвольно
(проверяется в `tests/test_ai.py`). `/api/status` возвращает `ai.enabled=true`,
`ai.mode=llm_with_verbatim_knowledge` и состояние каждого провайдера в `ai.routers`.
Живая веб-демонстрация доступна в preview-окне Arena.

## ИИ-слой в настоящем GitHub Actions

Провайдеры и их реальное поведение зафиксированы прогонами workflow
`Telegram - live acceptance test` (комментарии публикуются в PR #2):

- **`llm7` (без ключа) отвечает.** Run `37065369385` (push `2ae590a`): `ai-check`,
  провайдер `llm7`, модель `GLM-5.3-Flash`, вопрос «Как почистить КАН Ультра?» —
  `layer=llm`, `action=answer`, `latency_ms=214`, остальные вопросы — честный резерв
  `rules_fallback` из-за ответа `429 Rate limit exceeded`.
- **Диагностика внешних моделей.** Run `37064447653` (push `a9f2697`):
  `https://text.pollinations.ai/openai` с реальным payload бота вернул валидный
  `choices[0].message.content` (`'ок'`), `https://api.llm7.io/v1` — HTTP 200 с
  обычными параметрами, а `models.github.ai` в этом окружении отвечает
  `HTTP 200 text/plain 'OK'` на любой путь (заглушка), поэтому GitHub Models здесь
  недоступны.
- **Цепочка провайдеров.** Run `37068228548` (push `daef365`) напечатал
  `Провайдеры: llm7/GLM-5.3-Flash → pollinations/openai`, состояние каждого
  провайдера и причину отказа (`429 Daily token quota exceeded`, `402`).
  Run `37068628257` (push `72f5827`, токен workflow передан в проверку) показал
  полную цепочку `github/openai/gpt-4o-mini → llm7/GLM-5.3-Flash →
  pollinations/openai` и точную причину по каждому звену:
  `bad_response detail='OK'` (заглушка models.github.ai), `429` (квота llm7),
  `402` (pollinations). Когда модели недоступны, бот отвечает на всех пяти
  вопросах правильно — `layer=rules_fallback`, то есть без выдумывания.
- Обычный CI (`Checks`) на каждом push завершается успешно: runs `37068233136`,
  `37068228590`, `37067733048`, `37067411898`, `37066900377`, `37068628253`.
- Живой polling в этом прогоне: job `Manual live Telegram test` стартовал
  `2026-10-02T21:46Z`, сессия 90 минут — бот доступен по ссылке ниже в это время.

Важно: бесплатные keyless-шлюзы не дают гарантий и лимитированы. Для стабильной работы
достаточно добавить секрет `LLM_API_KEY` (любой OpenAI-совместимый провайдер) или
использовать `LLM_PROVIDER=github` в среде, где GitHub Models доступны — код и
проверки при этом не меняются, а при недоступности модели бот честно отвечает по базе.

## Telegram: что проверено

- Реальное подключение подтверждено: https://t.me/TestBotKovaleuski_bot,
  job `Prepare and verify Telegram` (getMe, getWebhookInfo, setMyCommands, getUpdates),
  публичная ссылка и имя бота совпадают с ожидаемыми.
- Живые сессии polling запускаются workflow: `telegram-demo.yml` (приёмочная сессия
  5–348 минут) и `telegram-host.yml` (постоянный хостинг после мержа в `main`,
  сессии по 5ч45м каждые 6 часов, общая concurrency-группа — два polling-процесса
  никогда не пересекаются).
- Содержимое ответов в Telegram формируется тем же `AIAssistant`, что проверен выше
  (сетевой слой Telegram протестирован через `httpx.MockTransport`).

**Не проверено автоматически:** живой диалог человека с ботом в Telegram — для этого
нужно отправить сообщение боту от аккаунта пользователя. Откройте ссылку, отправьте
`/start` и вопросы из таблицы выше; ответы берутся из базы дословно, поэтому проверка
сводится к факту доставки сообщения.

## Что осталось за рамками

- Docker-образ не собирался в этом окружении (`docker` недоступен), синтаксис
  `Dockerfile` и `docker-compose.yml` проверен вручную.
- Расписание `telegram-host.yml` начнёт действовать только после мержа ветки в `main`:
  GitHub запускает schedule только для workflow из ветки по умолчанию.
- Промышленной эксплуатации нет: прототип без CRM, заявок и авторизации демо.
