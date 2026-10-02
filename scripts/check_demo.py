"""Optional real-browser acceptance test for a running demo (desktop + mobile)."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from playwright.sync_api import expect, sync_playwright


def main() -> None:
    parser = argparse.ArgumentParser(description="Проверить живое демо в Chromium")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--screenshots", type=Path)
    args = parser.parse_args()
    if args.screenshots:
        args.screenshots.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        executable = os.getenv("BROWSER_EXECUTABLE")
        browser = p.chromium.launch(
            executable_path=executable or None,
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage"],
        )
        page = browser.new_page(viewport={"width": 1440, "height": 1050}, reduced_motion="reduce")
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("console", lambda msg: errors.append(msg.text) if msg.type == "error" else None)
        page.goto(args.url, wait_until="networkidle")
        expect(page.locator(".message.info")).to_have_count(1)
        expected = page.request.get(f"{args.url}/api/knowledge").json()["entries"]
        expected_by_id = {entry["id"]: entry for entry in expected}
        for index, entry_id in enumerate(
            ["kan_ultra_maintenance", "kit_frequency", "station_overflow", None]
        ):
            with page.expect_response(lambda res: res.url.endswith("/api/chat")) as result:
                page.locator("[data-question]").nth(index).click()
            reply = result.value.json()
            assert reply["entry_id"] == entry_id
            if entry_id:
                assert reply["text"] == expected_by_id[entry_id]["answer"]
                expect(page.locator(".message.answer .message-bubble").last).to_have_text(
                    reply["text"]
                )
            else:
                expect(page.locator(".message.missing .message-bubble")).to_contain_text(
                    "станции Тверь"
                )
                expect(page.locator(".message.missing .message-source")).to_have_count(0)
            expect(page.locator("#send-button")).to_be_enabled()
        if args.screenshots:
            page.screenshot(path=str(args.screenshots / "desktop.png"), full_page=True)

        page.locator("#reset-chat").click()
        expect(page.locator(".message")).to_have_count(1)
        page.locator("#question").fill("Как самому обслужить станцию?")
        page.locator("#question").press("Enter")
        expect(page.locator(".message.clarify")).to_have_count(1)
        page.locator(".clarification-options button").filter(has_text="КАН Ультра").click()
        expect(page.locator(".message.answer .message-bubble")).to_contain_text(
            "https://youtu.be/WW1Hqh3_WNk"
        )
        page.locator("#tab-knowledge").click()
        expect(page.locator(".knowledge-card")).to_have_count(len(expected))
        page.locator("#telegram-nav").click()
        expect(page.locator("#telegram-state")).to_have_text("Токен не настроен")
        expect(page.locator("#telegram-link")).to_be_hidden()
        assert not errors, errors

        # A transport failure must not be presented as an answer from the KB.
        page.route("**/api/chat", lambda route: route.abort())
        page.locator("#tab-dialog").click()
        page.locator("#question").fill("Как почистить КАН?")
        page.locator("#question").press("Enter")
        expect(page.locator(".message.error")).to_contain_text("техническая ошибка")
        expect(page.locator("#send-button")).to_be_enabled()
        page.unroute("**/api/chat")

        context = browser.new_context(
            viewport={"width": 375, "height": 812}, is_mobile=True, reduced_motion="reduce"
        )
        mobile = context.new_page()
        mobile.goto(args.url, wait_until="networkidle")
        expect(mobile.locator(".message.info")).to_have_count(1)
        mobile.locator(".unknown-example").click()
        expect(mobile.locator(".message.missing")).to_contain_text("станции Тверь")
        assert mobile.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
        expect(mobile.locator("#question")).to_be_visible()
        if args.screenshots:
            mobile.screenshot(path=str(args.screenshots / "mobile.png"), full_page=True)
        browser.close()
        print(
            "PASS: 4 сценария, уточнение, сброс, база, статус Telegram, ошибка сети, desktop/mobile"
        )


if __name__ == "__main__":
    main()
