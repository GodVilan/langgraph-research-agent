"""The published spend table reads a fixed window, not the last 30 days (D-059)."""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def trace(ts: str, env: str = "development", notional: float = 0.001) -> SimpleNamespace:
    usage = {"input_tokens": 1000, "output_tokens": 50, "cost_usd_notional": notional}
    return SimpleNamespace(
        timestamp=datetime.fromisoformat(ts).replace(tzinfo=UTC),
        environment=env,
        metadata={"usage": usage},
    )


# Either side of both window edges, and well after the date the old rolling window would have
# lost the Phase 4 run (2026-09-22 + 30 days).
STORE = [
    trace("2026-08-25T12:00:00", "integration-test"),  # August test traffic: out
    trace("2026-08-31T23:59:59"),  # a second before the window: out
    trace("2026-09-01T00:00:00"),  # the first instant: in
    trace("2026-09-22T18:00:00"),  # the Phase 4 traced run: in
    trace("2026-09-24T23:59:59"),  # the last second: in
    trace("2026-09-25T00:00:00"),  # the end, exclusive: out
    trace("2026-09-26T05:25:43", "integration-test"),  # D-058's mis-aimed test traces: out
    trace("2026-11-15T09:00:00"),  # long after: out
]


class FakeClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.api = SimpleNamespace(trace=SimpleNamespace(list=self.list))

    def list(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        # Return everything, as a server ignoring `to_timestamp` would: the filter must hold.
        return SimpleNamespace(data=STORE if kwargs["page"] == 1 else [])


def frozen_clock(monkeypatch: pytest.MonkeyPatch, module: Any, now: datetime) -> None:
    class Frozen(datetime):
        @classmethod
        def now(cls, tz: Any = None) -> datetime:  # type: ignore[override]
            return now

    monkeypatch.setattr(module, "datetime", Frozen)


@pytest.mark.parametrize(
    "now",
    [datetime(2026, 9, 28, tzinfo=UTC), datetime(2026, 11, 15, tzinfo=UTC)],
    ids=["before-2026-10-22", "after-2026-10-22"],
)
def test_the_published_table_is_the_same_on_any_day(
    now: datetime, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import scripts.budget_from_traces as budget

    client = FakeClient()
    monkeypatch.setattr(budget, "get_client", lambda: client)
    frozen_clock(monkeypatch, budget, now)
    monkeypatch.setattr(sys, "argv", ["budget_from_traces.py", "--dry-run"])
    assert budget.main() == 0
    out = capsys.readouterr().out
    assert "2026-09-01 to 2026-09-24 (UTC, inclusive)" in out
    assert "| `development` | 3 |" in out  # 09-01 00:00, 09-22, 09-24 23:59:59
    assert "integration-test" not in out  # neither August's nor D-058's
    assert "generated" not in out  # no clock in the block
    first = client.calls[0]
    assert first["from_timestamp"] == datetime(2026, 9, 1, tzinfo=UTC)
    assert first["to_timestamp"] == datetime(2026, 9, 25, tzinfo=UTC)


def test_two_days_render_byte_identical_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    import scripts.budget_from_traces as budget
    from scripts.trace_window import PUBLISHED_WINDOW, label, within

    blocks = []
    for now in (datetime(2026, 9, 28, tzinfo=UTC), datetime(2027, 3, 1, tzinfo=UTC)):
        frozen_clock(monkeypatch, budget, now)
        rows = budget.summarise(within(STORE, PUBLISHED_WINDOW))
        blocks.append(budget.render(rows, label(), 3))
    assert blocks[0] == blocks[1]


def test_a_rolling_window_is_never_written(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import scripts.budget_from_traces as budget

    monkeypatch.setattr(budget, "get_client", FakeClient)
    monkeypatch.setattr(sys, "argv", ["budget_from_traces.py", "--days", "30"])
    assert budget.main() == 2
    assert "Refusing to write a rolling window" in capsys.readouterr().err


def test_reconcile_reads_the_same_window(monkeypatch: pytest.MonkeyPatch) -> None:
    import scripts.reconcile_cost as reconcile

    client = FakeClient()
    monkeypatch.setattr(reconcile, "get_client", lambda: client)
    from scripts.trace_window import PUBLISHED_WINDOW

    kept = reconcile.fetch_traces(PUBLISHED_WINDOW)
    assert len(kept) == 3
    assert client.calls[0]["to_timestamp"] == PUBLISHED_WINDOW[1]
