"""Shared fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pydantic import SecretStr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import BudgetLimits, GraphLimits, RetrievalSettings, Settings
from src.observability.config import ObservabilitySettings


@pytest.fixture
def settings() -> Settings:
    """Deterministic settings with generous ceilings; individual tests tighten them."""
    return Settings(
        google_api_key="test-key",  # type: ignore[arg-type]
        budget=BudgetLimits(
            max_input_tokens=1_000_000,
            max_output_tokens=1_000_000,
            max_notional_cost_usd=1.0,
            max_wall_clock_s=3600.0,
            max_tool_calls=100,
            max_llm_calls=100,
        ),
        graph=GraphLimits(max_refinements=2, max_sub_questions=4, recursion_limit=25),
        retrieval=RetrievalSettings(),
    )


@pytest.fixture(autouse=True)
def _isolate_settings_cache(monkeypatch: pytest.MonkeyPatch, settings: Settings) -> None:
    """Make ``get_settings()`` return the test settings everywhere it is called.

    Nodes call ``get_settings()`` directly for limits, so without this the real .env would
    leak into tests.
    """
    for module in (
        "src.config",
        "src.agent.llm",
        "src.agent.nodes.plan",
        "src.agent.nodes.critique",
        "src.agent.nodes.validate_input",
        "src.agent.nodes.finalize",
        "src.agent.graph",
        "src.agent.runner",
    ):
        monkeypatch.setattr(f"{module}.get_settings", lambda: settings, raising=False)


@pytest.fixture(autouse=True)
def _isolate_observability(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the test suite out of the real Langfuse project.

    ``run_query`` opens a Langfuse root span unconditionally, and the observability
    settings are read from ``.env`` by a *separate* settings object that
    ``_isolate_settings_cache`` does not cover. With a key pair in ``.env``, every test
    that called ``run_query`` shipped a trace built from ``tests/fakes.py`` usage — round
    token counts and no cost — into the same `development` project that ``make budget``
    reads.

    That is how 187 of 214 traces came to be synthetic: the token columns summed fake
    traffic while the cost column summed only the six real runs, producing a blended rate
    below the input-only price. Two populations in one table (DECISIONS D-021).

    Tests that exercise the tracing code do so against recorders they inject themselves;
    none of them needs a live backend.
    """
    from src.observability import config as obs_config
    from src.observability import langfuse as lf

    disabled = ObservabilitySettings(
        langfuse_public_key=SecretStr(""),
        langfuse_secret_key=SecretStr(""),
        otlp_endpoint="",
    )
    obs_config.get_observability_settings.cache_clear()
    monkeypatch.setattr(obs_config, "get_observability_settings", lambda: disabled)
    monkeypatch.setattr(lf, "get_observability_settings", lambda: disabled, raising=False)
    # get_client() memoises across tests, so patching the settings alone is not enough:
    # a client built by an earlier test would survive into this one.
    monkeypatch.setattr(lf, "_CLIENT", None, raising=False)
    monkeypatch.setattr(lf, "_CLIENT_TRIED", False, raising=False)
