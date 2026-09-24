"""`make space-secrets` reads deploy credentials only from a file git will not commit."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.space_secrets import assert_gitignored

REPO = Path(__file__).resolve().parent.parent


def test_the_default_deploy_env_file_is_gitignored() -> None:
    assert_gitignored(REPO / ".env.deploy")  # raises if .gitignore stops covering it


def test_a_file_git_would_commit_is_refused() -> None:
    with pytest.raises(SystemExit, match="not gitignored"):
        assert_gitignored(REPO / "deploy-credentials.txt")  # no ignore rule covers this


def test_a_tracked_file_is_refused_even_if_a_pattern_matches() -> None:
    with pytest.raises(SystemExit, match="not gitignored"):
        assert_gitignored(REPO / "README.md")
