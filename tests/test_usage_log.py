"""Usage recording is structural: a model call through the wrapper leaves a log row.

The D-046 reconciliation found 76% of Google's billed prompt tokens recorded nowhere — eval
construction, the necessity checks and the probes all discarded the `Usage` the wrapper handed
them. The fix puts the record inside the wrapper, where a caller cannot drop it, and this file
checks both halves: the row appears, and nothing in the codebase builds a model or calls
Gemini's HTTP API around the wrapper.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from src.config import Settings

REPO = Path(__file__).resolve().parent.parent


def fake_model(**usage: int) -> GenericFakeChatModel:
    message = AIMessage(
        content="An answer.",
        usage_metadata={
            "input_tokens": usage.get("input", 1000),
            "output_tokens": usage.get("output", 50),
            "total_tokens": usage.get("input", 1000) + usage.get("output", 50),
            "input_token_details": {"cache_read": usage.get("cached", 0)},
        },
    )
    return GenericFakeChatModel(messages=iter([message]))


def rows(settings: Settings) -> list[dict[str, object]]:
    if not settings.usage_log.exists():
        return []
    lines = settings.usage_log.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


class TestEveryCallLeavesARow:
    async def test_a_call_through_the_wrapper_is_logged(self, settings: Settings) -> None:
        from src.agent.llm import USAGE_ACTIVITY, call_text

        token = USAGE_ACTIVITY.set("test-activity")
        try:
            await call_text("system", "user", model=fake_model(input=1000, output=50, cached=400))
        finally:
            USAGE_ACTIVITY.reset(token)
        (row,) = rows(settings)
        assert row["activity"] == "test-activity"
        assert row["input_tokens"] == 1000
        assert row["cached_input_tokens"] == 400
        assert row["output_tokens"] == 50
        expected = (600 * 0.30 + 400 * 0.03 + 50 * 2.50) / 1_000_000
        assert row["notional_cost_usd"] == pytest.approx(expected)

    async def test_a_caller_that_discards_usage_still_leaves_a_row(
        self, settings: Settings
    ) -> None:
        """What construction did: `drafted, _ = await call_structured(...)`."""
        from src.agent.llm import call_text

        _, _ = await call_text("system", "user", model=fake_model())
        assert len(rows(settings)) == 1

    async def test_every_attempt_is_logged(self, settings: Settings) -> None:
        from src.agent.llm import call_text

        for _ in range(3):
            await call_text("s", "u", model=fake_model())
        assert len(rows(settings)) == 3

    def test_the_suite_never_writes_the_real_log(self, settings: Settings) -> None:
        assert settings.usage_log != REPO / ".usage" / "llm_usage.jsonl"


class TestNothingCallsAroundTheWrapper:
    """The mechanism is only a guarantee if there is no second way to call the model."""

    SCANNED = ("src", "evals", "scripts")

    def files(self) -> list[Path]:
        out: list[Path] = []
        for top in self.SCANNED:
            for path in (REPO / top).rglob("*.py"):
                if "baselines" in path.parts:  # published v2.1 code, run unmodified (D-033)
                    continue
                out.append(path)
        return out

    def test_only_the_wrapper_builds_a_chat_model(self) -> None:
        builders = re.compile(r"\b(init_chat_model|ChatGoogleGenerativeAI)\(")
        offenders = [
            str(p.relative_to(REPO))
            for p in self.files()
            if p != REPO / "src" / "agent" / "llm.py" and builders.search(p.read_text())
        ]
        assert not offenders, f"models built around the usage log: {offenders}"

    def test_direct_gemini_http_calls_record_usage(self) -> None:
        offenders = [
            str(p.relative_to(REPO))
            for p in self.files()
            if "generativelanguage.googleapis.com" in p.read_text()
            and "record_usage(" not in p.read_text()
        ]
        assert not offenders, f"Gemini called over HTTP without a usage-log row: {offenders}"

    def test_a_direct_ainvoke_on_a_model_is_logged(self) -> None:
        """`model.ainvoke` returns a message whose usage nobody logs unless the caller passes it
        through `usage_from_message`; the graph's own `ainvoke` is not a model call."""
        offenders = []
        for p in self.files():
            text = p.read_text()
            model_calls = [
                line for line in text.splitlines() if ".ainvoke(" in line and "graph." not in line
            ]
            if p.name == "llm.py" or not model_calls:
                continue
            if "usage_from_message(" not in text:
                offenders.append(str(p.relative_to(REPO)))
        assert not offenders, f"direct model calls with no usage-log row: {offenders}"
