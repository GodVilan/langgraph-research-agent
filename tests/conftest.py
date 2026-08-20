"""Shared fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import BudgetLimits, GraphLimits, RetrievalSettings, Settings


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
