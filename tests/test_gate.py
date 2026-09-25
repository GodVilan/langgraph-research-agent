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

    def test_a_partial_sheet_is_refused_not_gated(self, tmp_path: Path) -> None:
        """D-053: a sheet assembled after q1 (q3 pending) scored 49 of 69 and the gate reported
        correct answers 0 vs 15 as a regression. An unfinished input exits 2 — and 2 is not 1,
        so a refusal can never pass for the gate firing."""
        import json
        import subprocess

        sheet = json.loads(self.SHEET.read_text(encoding="utf-8"))
        for item in [i for i in sheet["scores"] if i.startswith("sp-")][:20]:
            del sheet["scores"][item]
        partial = tmp_path / "partial.json"
        partial.write_text(json.dumps(sheet), encoding="utf-8")
        result = subprocess.run(
            [sys.executable, "-m", "evals.gate", "--run", str(self.RUN), "--sheet", str(partial)],
            cwd=self.ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 2
        assert "covers 49 of 69 items" in result.stderr

    def test_an_unreadable_sheet_exits_2_not_1(self, tmp_path: Path) -> None:
        """Exit 1 means "regression"; a crash on bad input must not look like one."""
        import subprocess

        bad = tmp_path / "bad.json"
        bad.write_text("{not json", encoding="utf-8")
        code = subprocess.run(
            [sys.executable, "-m", "evals.gate", "--run", str(self.RUN), "--sheet", str(bad)],
            cwd=self.ROOT,
            capture_output=True,
        ).returncode
        assert code == 2

    def test_gated_values_refuses_a_partial_sheet_on_every_path(self) -> None:
        """`--derive-baseline` goes through `gated_values` too; the refusal lives there."""
        from evals.gate import IncompleteSheetError, gated_values
        from evals.run_set import RunRecord
        from evals.schema import EvalSet
        from evals.scoring import ScoreSheet

        evalset = EvalSet.read(self.ROOT / "evals/datasets/phase4.json")
        record = RunRecord.model_validate_json(self.RUN.read_text(encoding="utf-8"))
        sheet = ScoreSheet.load(self.SHEET)
        sheet.scores.pop(next(iter(sheet.scores)))
        with pytest.raises(IncompleteSheetError):
            gated_values(evalset, record, sheet)

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


R = "evals/runs/v3_de699d68{}.json"
S = "evals/runs/scores_luna-low_de699d68_all{}.json"
COMMITTED_BASELINES = {
    # Phase 4, unpinned: r1-r3.
    "evals/baseline_metrics.json": [(R.format(t), S.format(t)) for t in ("", "_r2", "_r3")],
    # The shipped configuration, pinned: three draws (D-044).
    "evals/baseline_metrics_pinned.json": [
        (R.format(t), S.format(t)) for t in ("_pinned", "_pinned_r2", "_pinned_r3")
    ],
}


class TestDerivationRule:
    @pytest.mark.parametrize("baseline_path", sorted(COMMITTED_BASELINES))
    def test_the_rule_reproduces_each_committed_baseline(self, baseline_path: str) -> None:
        """Baselines are never hand-edited: each committed one must be regenerated exactly by
        the one derivation function from the runs it names. The Phase 4 baseline stated its
        rule but no committed code produced it until this test existed."""
        import json
        from pathlib import Path

        from evals.gate import derive_baseline, gated_values
        from evals.run_set import RunRecord
        from evals.schema import EvalSet
        from evals.scoring import ScoreSheet

        runs = COMMITTED_BASELINES[baseline_path]
        evalset = EvalSet.read(Path("evals/datasets/phase4.json"))
        committed = json.loads(Path(baseline_path).read_text())
        records = [RunRecord.model_validate_json(Path(r).read_text()) for r, _ in runs]
        per_run = [
            gated_values(evalset, rec, ScoreSheet.load(Path(sh)))
            for rec, (_, sh) in zip(records, runs, strict=True)
        ]
        derived = derive_baseline(per_run, records, committed, [Path(r) for r, _ in runs])
        assert set(derived["metrics"]) == set(committed["metrics"])
        for name, ref in committed["metrics"].items():
            got = derived["metrics"][name]
            assert got["value"] == pytest.approx(ref["value"])
            assert got["tolerance"] == pytest.approx(ref["tolerance"])
            assert [float(x) for x in got["runs"]] == [float(x) for x in ref["runs"]]

    def test_outcome_tolerance_is_floored_retrieval_is_not(self) -> None:
        """Three equal draws of an outcome metric are a small-sample artifact while generation
        and judging stay unpinned: the unpinned spread is the floor. Retrieval is deterministic
        by construction and keeps its own (zero) spread."""
        from pathlib import Path
        from types import SimpleNamespace

        from evals.gate import derive_baseline

        ref = {
            "metrics": {
                "correct_answers": {"value": 15.7, "tolerance": 1.0},
                "hallucinated_refusals": {"value": 25.3, "tolerance": 2.0},
                "factual_recall@5": {"value": 0.267, "tolerance": 0.05},
            }
        }
        rec = SimpleNamespace(set_sha256="s", corpus_sha256="c", scope_classifier_pinned=True)
        runs = [
            {"correct_answers": 15.0, "hallucinated_refusals": h, "factual_recall@5": 0.267}
            for h in (27.0, 23.0, 25.0)
        ]
        out = derive_baseline(runs, [rec] * 3, ref, [Path("a"), Path("b"), Path("c")])  # type: ignore[list-item]
        m = out["metrics"]
        assert m["correct_answers"]["tolerance"] == 1.0  # own 0, floored
        assert m["correct_answers"]["tolerance_basis"].startswith("floor")
        assert m["hallucinated_refusals"]["tolerance"] == 4.0  # own 4 > floor 2
        assert m["factual_recall@5"]["tolerance"] == 0.0  # retrieval: own, not floored

    def test_one_run_carries_the_reference_spread(self) -> None:
        from pathlib import Path
        from types import SimpleNamespace

        from evals.gate import derive_baseline

        ref = {"metrics": {"hallucinated_refusals": {"value": 25.3, "tolerance": 2.0}}}
        rec = SimpleNamespace(set_sha256="s", corpus_sha256="c", scope_classifier_pinned=True)
        out = derive_baseline([{"hallucinated_refusals": 27.0}], [rec], ref, [Path("x")])  # type: ignore[list-item]
        assert out["metrics"]["hallucinated_refusals"]["tolerance"] == 2.0
        assert "carried" in out["derivation"]

    def test_runs_of_different_configurations_are_refused(self) -> None:
        from pathlib import Path
        from types import SimpleNamespace

        from evals.gate import derive_baseline

        ref = {"metrics": {"m": {"value": 1.0, "tolerance": 0.0}}}
        a = SimpleNamespace(set_sha256="s", corpus_sha256="c", scope_classifier_pinned=True)
        b = SimpleNamespace(set_sha256="s", corpus_sha256="c", scope_classifier_pinned=None)
        with pytest.raises(ValueError, match="one configuration"):
            derive_baseline([{"m": 1.0}, {"m": 1.0}], [a, b], ref, [Path("x"), Path("y")])  # type: ignore[list-item]


class TestStricterOfBaselines:
    def test_each_metric_takes_the_tighter_bound(self) -> None:
        from evals.gate import evaluate, strictest

        phase4 = {
            "metrics": {
                "hallucinated_refusals": {"value": 25.333, "tolerance": 2.0},  # ceiling 27.333
                "correct_answers": {"value": 15.667, "tolerance": 1.0},  # floor 14.667
            }
        }
        pinned1 = {
            "metrics": {
                "hallucinated_refusals": {"value": 27.0, "tolerance": 2.0},  # ceiling 29
                "correct_answers": {"value": 15.0, "tolerance": 1.0},  # floor 14
            }
        }
        combined = strictest([phase4, pinned1])
        assert combined["metrics"]["hallucinated_refusals"]["value"] == pytest.approx(27.333)
        assert combined["metrics"]["correct_answers"]["value"] == pytest.approx(14.667)
        # 28 refusals passes the loosened single-draw baseline but not the stricter bound.
        assert not evaluate({"hallucinated_refusals": 28.0}, pinned1)
        assert evaluate({"hallucinated_refusals": 28.0}, combined)
