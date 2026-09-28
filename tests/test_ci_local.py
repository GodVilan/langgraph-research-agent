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

    def test_the_workflow_is_read_from_the_export_not_the_checkout(self) -> None:
        """A relative path, resolved against the export: the commit's own ci.yml is what runs."""
        import inspect

        from scripts import ci_local

        assert not ci_local.WORKFLOW.is_absolute()
        assert "export_dir / WORKFLOW" in inspect.getsource(ci_local.main)


class TestIntegrationNeverTouchesTheDevelopersStack:
    """`make ci-local INTEGRATION=1` runs `docker compose … up -d`. On the default project that
    would recreate the developer's stack with a MinIO that cannot write its root-owned volume
    and put test traces into the store the spend table reads (D-057, D-058)."""

    def test_the_default_project_is_refused(self) -> None:
        from scripts.ci_local import integration_refusal

        why = integration_refusal("arxiv-agent-langfuse", "arxiv-agent-langfuse", [])
        assert "default project" in why

    def test_a_project_that_already_has_volumes_is_refused(self) -> None:
        from scripts.ci_local import integration_refusal

        why = integration_refusal(
            "ci-local-integration-x", "arxiv-agent-langfuse", ["ci-local-integration-x_minio"]
        )
        assert "already has volumes" in why

    def test_a_fresh_throwaway_project_is_accepted(self) -> None:
        from scripts.ci_local import integration_refusal

        volumes = ["arxiv-agent-langfuse_langfuse-minio", "arxiv-agent-langfuse_langfuse-postgres"]
        assert integration_refusal("ci-local-integration-x", "arxiv-agent-langfuse", volumes) == ""

    def test_the_compose_file_default_is_what_is_protected(self) -> None:
        from scripts.ci_local import compose_default_project

        assert compose_default_project(REPO) == "arxiv-agent-langfuse"

    @pytest.mark.parametrize("collision", ["default", "volumes"])
    def test_run_integration_refuses_before_calling_docker(
        self, collision: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import uuid

        import scripts.ci_local as ci

        monkeypatch.setattr(uuid, "uuid4", lambda: uuid.UUID(int=0xDEADBEEF))
        project = "ci-local-integration-00000000"
        monkeypatch.setattr(
            ci,
            "compose_default_project",
            lambda export_dir: project if collision == "default" else "arxiv-agent-langfuse",
        )
        monkeypatch.setattr(
            ci, "docker_volumes", lambda: [f"{project}_minio"] if collision == "volumes" else []
        )

        def no_subprocess(*args: object, **kwargs: object) -> None:
            raise AssertionError(f"called {args[0]!r} after a refusal")

        monkeypatch.setattr(ci.subprocess, "run", no_subprocess)
        with pytest.raises(SystemExit, match="refusing the integration step"):
            ci.run_integration(tmp_path, {}, {}, {"run": "true"})
