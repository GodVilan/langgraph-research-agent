"""Run CI's own commands in a clean export of the repository, before pushing.

    make ci-local              # a `git archive` of HEAD
    make ci-local WORKTREE=1   # what `git add -A` would stage, over HEAD (nothing is modified)

Why it exists (DECISIONS D-056): from run #9 to the fix, every CI run failed at the unit-test
step, on two defects that passed on this machine only because it held files the repository does
not — `CLAUDE.md` (untracked) and the FAISS index (gitignored). The regression gate, the dataset
re-verification and the integration suite were later steps of that job and never executed on
GitHub; every green meanwhile was local. A local run proves nothing about CI unless it runs
**from what git holds**, with **CI's commands**, in **CI's environment**. So this script:

* exports the tree with `git archive` into a temporary directory — untracked and ignored files
  cannot reach it, which is the point — and commits it there as a one-commit repository, as
  `actions/checkout` does; `WORKTREE=1` archives what `git add -A` would stage, built in a
  temporary index (the real index and refs are untouched), and names the files not yet in HEAD;
* reads `.github/workflows/ci.yml` and runs each step's `run:` block verbatim, with
  `bash -eo pipefail` as GitHub does, the workflow's `env:` at every level, and otherwise a
  minimal environment — no `.env`, no shell exports, no API keys;
* asserts the code under test is the export's, not this checkout's editable install;
* reports every step it does not run, and why. Installing (this repo's venv stands in for
  `pip install -e ".[dev]"`), the Docker-based integration suite, and the from-scratch
  clean-install job are NOT RUN here, and are listed as such — never silently skipped.

Exits non-zero if any executed step fails.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]  # no stubs; scripts/ is outside `make type`

REPO = Path(__file__).resolve().parent.parent
WORKFLOW = REPO / ".github" / "workflows" / "ci.yml"
VENV_BIN = REPO / ".venv" / "bin"

# Steps not run here, by (job, step-name prefix), each with the reason printed in the report.
# Job-specific entries first: the first match wins.
NOT_RUN = {
    ("integration", "Integration"): "needs Docker and a six-container Langfuse stack",
    ("clean-install", ""): "installs from scratch into a fresh environment (network, minutes)",
    ("*", "Install"): "this repo's .venv stands in for `pip install -e .[dev]`",
}


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.strip()


def source_rev(worktree: bool) -> tuple[str, str]:
    """The commit to export, and a label for it.

    ``worktree``: a commit object of everything ``git add -A`` would stage — tracked changes and
    untracked files that are not ignored — built in a temporary index file, so the real index,
    HEAD and every ref are untouched. That is what the next commit would hold if everything were
    added; the untracked files it includes are listed, since a commit that leaves them out is a
    different tree."""
    head = git("rev-parse", "HEAD")
    if not worktree:
        return head, f"HEAD {head[:7]}"
    with tempfile.TemporaryDirectory() as tmp:
        env = {**os.environ, "GIT_INDEX_FILE": str(Path(tmp) / "index")}

        def run(*args: str) -> str:
            return subprocess.run(
                ["git", *args], cwd=REPO, env=env, capture_output=True, text=True, check=True
            ).stdout.strip()

        run("read-tree", "HEAD")
        run("add", "-A")
        tree = run("write-tree")
        added = run("diff", "--cached", "--name-only", "--diff-filter=A", "HEAD").splitlines()
        commit = run("commit-tree", tree, "-p", head, "-m", "ci-local worktree")
    if added:
        print("WORKTREE: files not in HEAD, included (they must be committed for CI to see them):")
        for f in added:
            print(f"    {f}")
    return commit, f"working tree (as `git add -A` would stage it) over HEAD {head[:7]}"


def export(rev: str, dest: Path) -> None:
    """`git archive` the commit, then make the export a one-commit repository of exactly those
    files — what `actions/checkout` gives the runner. A bare archive has no `.git`, so any test
    that asks git a question (is `.env.deploy` ignored?) would fail here and pass on GitHub.
    `-f` adds every file: all of them are tracked by construction, including any tracked file
    that happens to match an ignore pattern."""
    archive = subprocess.run(["git", "archive", rev], cwd=REPO, capture_output=True, check=True)
    subprocess.run(["tar", "-x", "-C", str(dest)], input=archive.stdout, check=True)
    for cmd in (
        ["init", "-q"],
        ["add", "-A", "-f"],
        ["-c", "user.email=ci-local@localhost", "-c", "user.name=ci-local", "commit", "-qm", rev],
    ):
        subprocess.run(["git", *cmd], cwd=dest, capture_output=True, check=True)


def environment(export_dir: Path, *levels: dict[str, Any]) -> dict[str, str]:
    git_dir = str(Path(shutil.which("git") or "/usr/bin/git").parent)
    env = {
        "PATH": os.pathsep.join([str(VENV_BIN), git_dir, "/usr/bin", "/bin", "/usr/sbin", "/sbin"]),
        "HOME": os.environ.get("HOME", "/tmp"),
        "LANG": "C.UTF-8",
        "TMPDIR": tempfile.gettempdir(),
        "CI": "true",
        # The export's own packages win over this checkout's editable install.
        "PYTHONPATH": str(export_dir),
    }
    for level in levels:
        env.update({k: str(v) for k, v in (level or {}).items()})
    return env


def skipped(job: str, step: str) -> str:
    for (j, prefix), why in NOT_RUN.items():
        if j in ("*", job) and step.startswith(prefix):
            return why
    return ""


def assert_code_is_exported(export_dir: Path, env: dict[str, str]) -> None:
    probe = "import src, evals; print(src.__file__); print(evals.__file__)"
    out = subprocess.run(
        ["python", "-c", probe], cwd=export_dir, env=env, capture_output=True, text=True
    )
    paths = out.stdout.split()
    if (
        out.returncode != 0
        or not paths
        or not all(Path(p).resolve().is_relative_to(export_dir.resolve()) for p in paths)
    ):
        raise SystemExit(
            f"refusing: the code under test is not the export's ({paths or out.stderr[:200]})"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--worktree", action="store_true")
    parser.add_argument("--keep", action="store_true", help="keep the export directory")
    args = parser.parse_args()

    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    rev, label = source_rev(args.worktree)
    export_dir = Path(tempfile.mkdtemp(prefix="ci-local-"))
    export(rev, export_dir)
    print(f"ci-local: {label}, exported to {export_dir}")
    assert_code_is_exported(export_dir, environment(export_dir, workflow.get("env") or {}))

    results: list[tuple[str, str, str]] = []
    for job_id, job in workflow["jobs"].items():
        for step in job["steps"]:
            if "run" not in step:
                continue
            name = str(step.get("name", step["run"].splitlines()[0]))
            why = skipped(job_id, name)
            if why:
                results.append((job_id, name, f"NOT RUN — {why}"))
                continue
            env = environment(
                export_dir, workflow.get("env") or {}, job.get("env") or {}, step.get("env") or {}
            )
            print(f"\n=== [{job_id}] {name}", flush=True)
            code = subprocess.run(
                ["bash", "--noprofile", "--norc", "-eo", "pipefail", "-c", step["run"]],
                cwd=export_dir,
                env=env,
            ).returncode
            results.append((job_id, name, "pass" if code == 0 else f"FAIL (exit {code})"))

    print(f"\nci-local summary — {label}")
    for job_id, name, outcome in results:
        print(f"  [{job_id:13}] {name:58} {outcome}")
    failed = [r for r in results if r[2].startswith("FAIL")]
    if not args.keep:
        shutil.rmtree(export_dir, ignore_errors=True)
    print(
        f"\n{'FAILED' if failed else 'green'}: {len(failed)} of "
        f"{sum(1 for r in results if not r[2].startswith('NOT RUN'))} executed steps failed"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
