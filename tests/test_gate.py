"""The regression gate must fire on an injected regression, and must not fire on the baseline.

A gate that has never fired has never been tested. Both directions are asserted: the
baseline passes itself, and a copy of the baseline run with a regression injected — every
retrieved chunk dropped, so Recall@5 falls to zero — fails with the metric named.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.gate import evaluate

BASELINE = {
    "metrics": {
        "factual_recall@5": {"value": 0.267, "tolerance": 0.05},
        "hallucinated_refusals": {"value": 7.0, "tolerance": 1.0},
        "attribute_correct_refusals": {"value": 8.0, "tolerance": 1.0},
    }
}


class TestGateFires:
    def test_the_baseline_passes_itself(self) -> None:
        assert evaluate({"factual_recall@5": 0.267, "hallucinated_refusals": 7.0}, BASELINE) == []

    def test_a_move_inside_the_measured_spread_is_not_a_regression(self) -> None:
        assert evaluate({"factual_recall@5": 0.23, "hallucinated_refusals": 8.0}, BASELINE) == []

    def test_a_retrieval_collapse_fails_and_names_the_metric(self) -> None:
        failures = evaluate({"factual_recall@5": 0.0}, BASELINE)

        assert len(failures) == 1
        assert failures[0].startswith("factual_recall@5")

    def test_more_hallucinated_refusals_is_a_regression(self) -> None:
        """Lower is better here; the direction is per metric, not global."""
        assert evaluate({"hallucinated_refusals": 10.0}, BASELINE)

    def test_fewer_hallucinated_refusals_is_not(self) -> None:
        assert evaluate({"hallucinated_refusals": 2.0}, BASELINE) == []

    def test_an_improvement_in_a_higher_is_better_metric_is_not_a_regression(self) -> None:
        assert evaluate({"factual_recall@5": 0.9}, BASELINE) == []

    def test_an_ungated_metric_is_ignored_not_failed(self) -> None:
        assert evaluate({"something_new": 0.0}, BASELINE) == []


class TestTheGateFiresEndToEnd:
    """The CLI, not just `evaluate` — against the committed baseline and a regressed copy.

    CI runs the same two assertions (`.github/workflows/ci.yml`). Having them here too means
    the demonstration is reproducible on any machine without a push, which matters because the
    carry was "demonstrate the CI gate failing" and a gate that has only ever passed has not
    been shown to fire.
    """

    ROOT = Path(__file__).resolve().parent.parent
    RUN = ROOT / "evals/runs/v3_de699d68.json"
    SHEET = ROOT / "evals/runs/scores_luna-low_de699d68_all.json"
    BASELINE = ROOT / "evals/baseline_metrics.json"

    def _gate(self, run: Path) -> int:
        import subprocess

        return subprocess.run(
            [sys.executable, "-m", "evals.gate", "--run", str(run), "--sheet", str(self.SHEET)],
            cwd=self.ROOT,
            capture_output=True,
        ).returncode

    def test_the_baseline_run_passes_its_own_gate(self) -> None:
        if not (self.BASELINE.exists() and self.RUN.exists()):
            pytest.skip("baseline or run artifact not committed")

        assert self._gate(self.RUN) == 0

    def test_an_injected_regression_fails_the_gate(self, tmp_path: Path) -> None:
        import json

        if not (self.BASELINE.exists() and self.RUN.exists()):
            pytest.skip("baseline or run artifact not committed")

        run = json.loads(self.RUN.read_text(encoding="utf-8"))
        for item in run["items"].values():
            item["retrieved"] = []  # retrieval collapse; nothing else changed
        regressed = tmp_path / "regressed.json"
        regressed.write_text(json.dumps(run), encoding="utf-8")

        assert self._gate(regressed) == 1, "the gate passed an injected regression"

    def test_the_gate_refuses_to_run_without_a_baseline(self, tmp_path: Path) -> None:
        """An underived tolerance is worse than no gate: exit 2, not a pass."""
        import subprocess

        code = subprocess.run(
            [
                sys.executable,
                "-m",
                "evals.gate",
                "--run",
                str(self.RUN),
                "--baseline",
                str(tmp_path / "absent.json"),
            ],
            cwd=self.ROOT,
            capture_output=True,
        ).returncode

        assert code == 2
