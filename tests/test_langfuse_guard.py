"""`make langfuse-up` refuses while the MinIO volume cannot be written by the pinned image."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# `ls -lnd` of the three paths, as read from the two real volumes on 2026-09-28.
ROOT_RUN_VOLUME = """drwxr-xr-x 4 0 0 4096 Aug 21 16:14 /data
drwxr-xr-x 4 0 0 4096 Aug 21 16:17 /data/langfuse
drwxr-xr-x 7 0 0 4096 Sep 22 23:23 /data/.minio.sys
"""
FRESH_CHAINGUARD_VOLUME = """drwxrwxrwx 4 0 0 4096 Sep 26 05:20 /data
drwxr-xr-x 2 65532 65532 4096 Sep 26 05:20 /data/langfuse
"""


def fake_docker(monkeypatch: pytest.MonkeyPatch, volume_exists: bool, ls: str) -> list[list[str]]:
    import scripts.langfuse_guard as guard

    calls: list[list[str]] = []

    def run(*args: str) -> subprocess.CompletedProcess[str]:
        calls.append(list(args))
        if args[:2] == ("volume", "inspect"):
            return subprocess.CompletedProcess(args, 0 if volume_exists else 1, "", "")
        if args[:2] == ("image", "inspect"):
            return subprocess.CompletedProcess(args, 0, "65532\n", "")
        if args[0] == "run":
            return subprocess.CompletedProcess(args, 0, ls, "")
        if args[0] == "compose":
            raise AssertionError("the guard must never start anything")
        return subprocess.CompletedProcess(args, 1, "", "unexpected")

    monkeypatch.setattr(guard, "docker", run)
    return calls


class TestWritability:
    def test_the_root_run_volume_is_unwritable_by_uid_65532(self) -> None:
        from scripts.langfuse_guard import unwritable

        rows = [
            (line.split()[0], int(line.split()[2]), int(line.split()[3]), line.split()[-1])
            for line in ROOT_RUN_VOLUME.splitlines()
        ]
        assert len(unwritable(65532, rows)) == 3

    def test_a_fresh_chainguard_volume_is_writable(self) -> None:
        """Root-owned `/data` is not the test: a fresh volume is root-owned *and* mode 777."""
        from scripts.langfuse_guard import unwritable

        rows = [
            (line.split()[0], int(line.split()[2]), int(line.split()[3]), line.split()[-1])
            for line in FRESH_CHAINGUARD_VOLUME.splitlines()
        ]
        assert unwritable(65532, rows) == []

    def test_a_root_image_can_write_anything(self) -> None:
        from scripts.langfuse_guard import unwritable

        assert unwritable(0, [("drwxr-xr-x", 0, 0, "/data")]) == []


class TestTheRefusal:
    def test_the_existing_root_owned_volume_is_refused_with_the_migration_steps(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        import scripts.langfuse_guard as guard

        calls = fake_docker(monkeypatch, volume_exists=True, ls=ROOT_RUN_VOLUME)
        assert guard.main() == 1
        err = capsys.readouterr().err
        assert "refusing `make langfuse-up`" in err
        assert "arxiv-agent-langfuse_langfuse-minio" in err
        assert "**Plan, in order:**" in err and "chown -R 65532:65532" in err
        # The volume was only ever mounted read-only.
        mounts = [c[c.index("-v") + 1] for c in calls if c[0] == "run"]
        assert mounts and all(m.endswith(":/data:ro") for m in mounts)

    def test_a_fresh_stack_with_no_volume_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import scripts.langfuse_guard as guard

        fake_docker(monkeypatch, volume_exists=False, ls="")
        assert guard.main() == 0

    def test_a_migrated_writable_volume_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import scripts.langfuse_guard as guard

        fake_docker(monkeypatch, volume_exists=True, ls=FRESH_CHAINGUARD_VOLUME)
        assert guard.main() == 0

    def test_the_protected_volume_follows_the_compose_project(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import yaml

        from scripts.langfuse_guard import COMPOSE, compose_minio

        compose: dict[str, Any] = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))
        monkeypatch.delenv("COMPOSE_PROJECT_NAME", raising=False)
        image, volume = compose_minio(compose)
        assert volume == "arxiv-agent-langfuse_langfuse-minio"
        assert image.startswith("cgr.dev/chainguard/minio")
        monkeypatch.setenv("COMPOSE_PROJECT_NAME", "throwaway")
        assert compose_minio(compose)[1] == "throwaway_langfuse-minio"

    def test_make_langfuse_up_runs_the_guard_before_compose(self) -> None:
        makefile = (REPO / "Makefile").read_text(encoding="utf-8")
        recipe = makefile.split("\nlangfuse-up:", 1)[1].split("\n\n", 1)[0]
        guard_at = recipe.index("scripts/langfuse_guard.py")
        assert guard_at < recipe.index("docker compose -f infra/docker-compose.langfuse.yml up")
