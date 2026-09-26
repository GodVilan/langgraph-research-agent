"""`make ci-local` and the workflow shape it mirrors (DECISIONS D-056)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

GATE_STEPS = (
    "Re-verify the eval dataset",
    "Regression gate — baseline passes",
    "Regression gate — an injected regression must fail it",
)


def workflow() -> dict[str, object]:
    return dict(yaml.safe_load((REPO / ".github/workflows/ci.yml").read_text(encoding="utf-8")))


class TestTheGateCannotBeSwitchedOffByAnotherJob:
    """From run #9 the unit tests failed first and every later step of the same job — the
    gate among them — was skipped on every run. The gate now has its own job."""

    def test_no_job_waits_on_another(self) -> None:
        jobs = workflow()["jobs"]
        assert isinstance(jobs, dict)
        assert all("needs" not in job for job in jobs.values())

    def test_the_gate_steps_live_in_their_own_job_and_run_even_after_a_failed_step(self) -> None:
        jobs = workflow()["jobs"]
        assert isinstance(jobs, dict)
        gate = {s.get("name"): s for s in jobs["gate"]["steps"]}
        for name in GATE_STEPS:
            assert name in gate, name
            assert gate[name].get("if") == "${{ !cancelled() }}", name
        assert not any(s.get("name") in GATE_STEPS for s in jobs["check"]["steps"])


class TestCiLocal:
    def test_the_environment_carries_no_keys_from_the_calling_shell(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from scripts.ci_local import environment

        monkeypatch.setenv("GOOGLE_API_KEY", "must-not-leak")
        monkeypatch.setenv("OPENAI_API_KEY", "must-not-leak")
        env = environment(tmp_path, {"LANGFUSE_PUBLIC_KEY": ""})
        assert "GOOGLE_API_KEY" not in env and "OPENAI_API_KEY" not in env
        assert env["LANGFUSE_PUBLIC_KEY"] == ""
        assert env["PYTHONPATH"] == str(tmp_path)  # the export's code, not the checkout's

    @pytest.mark.parametrize(
        ("job", "step", "reason"),
        [
            ("integration", "Integration tests against a real Langfuse", "Docker"),
            ("clean-install", "Install from pyproject.toml alone", "from scratch"),
            ("clean-install", "Import every module from the installed distribution", "scratch"),
            ("gate", "Install with dev extras", ".venv"),
        ],
    )
    def test_what_is_not_run_is_named_with_its_reason(
        self, job: str, step: str, reason: str
    ) -> None:
        from scripts.ci_local import skipped

        assert reason in skipped(job, step)

    @pytest.mark.parametrize("step", ["Tests", "Lint", "Type check", *GATE_STEPS])
    def test_ci_commands_are_not_skipped(self, step: str) -> None:
        from scripts.ci_local import skipped

        job = "gate" if step in GATE_STEPS else "check"
        assert skipped(job, step) == ""

    def test_every_workflow_step_is_deliberately_run_or_named(self) -> None:
        """A step added to the workflow must be placed on one list or the other; otherwise
        ci-local would run it without anyone deciding it can run here."""
        from scripts.ci_local import skipped

        run_here = {"Lint", "Type check", "Tests", *GATE_STEPS}
        jobs = workflow()["jobs"]
        assert isinstance(jobs, dict)
        for job_id, job in jobs.items():
            for step in job["steps"]:
                if "run" not in step:
                    continue
                name = step["name"]
                assert (name in run_here) != bool(skipped(job_id, name)), (job_id, name)
