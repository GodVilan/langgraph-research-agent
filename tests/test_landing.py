"""GET / is a static page that talks only to its own origin; GET /api is the JSON (D-061)."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest

from src.api.landing import EXAMPLE_QUESTION, PAGE, page
from src.config import Settings
from tests.api_harness import running
from tests.fakes import FakeRetrievalService
from tests.test_api import api_settings  # noqa: F401  (fixture)

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


class TestRoutes:
    async def test_root_is_the_page(self, api_settings: Settings) -> None:  # noqa: F811
        async with running(api_settings, FakeRetrievalService()) as (_, client):
            r = await client.get("/")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/html")
        assert EXAMPLE_QUESTION in r.text
        assert "__EXAMPLE_QUESTION__" not in r.text
        csp = r.headers["content-security-policy"]
        assert "default-src 'none'" in csp and "connect-src 'self'" in csp

    async def test_api_is_the_json_description(self, api_settings: Settings) -> None:  # noqa: F811
        async with running(api_settings, FakeRetrievalService()) as (_, client):
            r = await client.get("/api")
        body = r.json()
        assert r.status_code == 200
        assert "POST /query" in body["endpoints"] and "GET /" in body["endpoints"]
        assert EXAMPLE_QUESTION in body["example"]
        assert "LoRA" not in body["example"]


class TestThePageTalksOnlyToItsOwnOrigin:
    """Checked in the file, and enforced in the browser by the CSP the route sends."""

    def html(self) -> str:
        return page()[0]

    def test_every_fetch_is_a_same_origin_path(self) -> None:
        targets = re.findall(r"fetch\(\s*([^,)]+)", self.html())
        assert targets, "the page must call the API"
        for target in targets:
            literal = target.strip()
            assert re.fullmatch(r"""["']/[^/"'][^"']*["']""", literal), literal

    def test_nothing_is_loaded_from_anywhere(self) -> None:
        html = self.html()
        assert not re.search(r"<script[^>]*\ssrc=", html)
        assert not re.search(r"<link\b", html)
        assert not re.search(r"<(img|iframe|video|audio|source|object|embed)\b", html)
        assert "@import" not in html and "url(" not in html
        assert not re.search(r"\b(XMLHttpRequest|WebSocket|EventSource|sendBeacon|import\()", html)

    def test_the_csp_admits_exactly_the_page_s_own_inline_code(self) -> None:
        import base64
        import hashlib

        html, csp = page()
        for kind, body in re.findall(r"<(script|style)>(.*?)</\1>", html, re.DOTALL):
            digest = base64.b64encode(hashlib.sha256(body.encode()).digest()).decode()
            assert f"'sha256-{digest}'" in csp, f"{kind} would be blocked by its own CSP"
        assert "unsafe-inline" not in csp and "unsafe-eval" not in csp

    def test_server_text_is_never_parsed_as_html(self) -> None:
        html = self.html()
        assert "innerHTML" not in html and "insertAdjacentHTML" not in html
        assert "document.write" not in html


class TestOneExampleQuestion:
    """The page, /api, the README's curl (which `make smoke-live` runs) and the Space card
    carry the same question — the one verified to cite a source (D-061)."""

    def test_the_readme_curl_asks_it(self) -> None:
        from scripts.smoke_live import readme_question

        assert readme_question() == EXAMPLE_QUESTION

    def test_the_space_card_asks_it(self) -> None:
        from scripts.deploy_space import SPACE_CARD

        assert EXAMPLE_QUESTION in SPACE_CARD

    def test_the_page_file_carries_only_the_placeholder(self) -> None:
        """One copy of the question, in src/api/landing.py; the file carries a placeholder."""
        raw = PAGE.read_text(encoding="utf-8")
        assert '"__EXAMPLE_QUESTION__"' in raw and EXAMPLE_QUESTION not in raw


class TestSmokeLiveChecksTheLandingPage:
    def fake(self, monkeypatch: pytest.MonkeyPatch, page_text: str, api_example: str) -> None:
        import scripts.smoke_live as smoke

        html, csp = page()

        def get(url: str, **_: Any) -> httpx.Response:
            request = httpx.Request("GET", url)
            if url.endswith("/api"):
                return httpx.Response(200, json={"example": api_example}, request=request)
            return httpx.Response(
                200,
                text=page_text or html,
                headers={"content-type": "text/html", "content-security-policy": csp},
                request=request,
            )

        monkeypatch.setattr(smoke.httpx, "get", get)

    def test_it_passes_on_the_real_page(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import scripts.smoke_live as smoke

        self.fake(monkeypatch, "", EXAMPLE_QUESTION)
        smoke.check_landing("https://example.invalid")

    def test_it_fails_when_the_page_lacks_the_example(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import scripts.smoke_live as smoke

        self.fake(monkeypatch, "<html>What is LoRA?</html>", EXAMPLE_QUESTION)
        with pytest.raises(smoke.SmokeFailureError, match="landing-example"):
            smoke.check_landing("https://example.invalid")
