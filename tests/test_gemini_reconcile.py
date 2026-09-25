"""The Gemini reconciliation compares the bill with instrumentation over the *same* period."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def test_runs_after_the_billing_period_are_not_counted_as_measured() -> None:
    from scripts.gemini_reconcile import billing_period_end, run_artifacts

    end = billing_period_end()
    counted = {r["source"].removeprefix("run artifact ") for r in run_artifacts()}
    for path in (REPO / "evals" / "runs").glob("v3_de699d68*.json"):
        started = str(json.loads(path.read_text(encoding="utf-8")).get("started_at", ""))[:10]
        assert (path.name in counted) == (started <= end), path.name


def test_the_committed_reconciliation_is_what_the_script_builds() -> None:
    """Regenerating must not move the published gap — the artifact is the script's output."""
    from scripts.gemini_reconcile import run_artifacts

    committed = json.loads((REPO / "evals/runs/gemini_reconcile.json").read_text(encoding="utf-8"))
    runs = [s for s in committed["sources"] if s["source"].startswith("run artifact")]
    assert sorted(r["source"] for r in runs) == sorted(r["source"] for r in run_artifacts())
