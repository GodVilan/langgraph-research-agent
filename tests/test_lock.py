"""One hashed lock for the image, CI and the local venv, and a check that fails on drift (D-064).

The image used to `pip install .` against open lower bounds, so each rebuild resolved its own
set: the 2026-10-02 build came up with 22 packages at newer versions, 2 new ones and Python
3.13.16, and its traces lost their root. These tests hold the three installs to the lock and
prove the check fires — on a fake installed set, and on this interpreter's real one.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import ClassVar

import pytest
import yaml  # type: ignore[import-untyped]
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from scripts.lock_check import LockError, compare, lock_python, parse_lock, python_problems

REPO = Path(__file__).resolve().parent.parent
LOCK = REPO / "requirements.lock"
DEV_LOCK = REPO / "requirements-dev.lock"
DEV_IN = REPO / "requirements-dev.in"
LINUX = {"sys_platform": "linux"}
DARWIN = {"sys_platform": "darwin"}
H = "--hash=sha256:" + "0" * 64


def lock_file(tmp_path: Path, text: str, name: str = "requirements.lock") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


# ── Reading a lock ───────────────────────────────────────────────────────────


class TestParse:
    def test_exact_hashed_pins_across_continuations(self, tmp_path: Path) -> None:
        path = lock_file(tmp_path, f"# c\n--extra-index-url https://x\nFoo_Bar==1.2 \\\n    {H}\n")
        assert parse_lock(path) == {"foo-bar": "1.2"}

    def test_exclusive_markers_give_one_torch_per_platform(self, tmp_path: Path) -> None:
        path = lock_file(
            tmp_path,
            f'torch==2.14.0+cpu ; sys_platform == "linux" {H}\n'
            f'torch==2.14.0 ; sys_platform == "darwin" {H}\n',
        )
        assert parse_lock(path, LINUX) == {"torch": "2.14.0+cpu"}
        assert parse_lock(path, DARWIN) == {"torch": "2.14.0"}

    def test_follows_an_include(self, tmp_path: Path) -> None:
        lock_file(tmp_path, f"a==1 {H}\n")
        dev = lock_file(tmp_path, f"-r requirements.lock\nb==2 {H}\n", "dev.lock")
        assert parse_lock(dev) == {"a": "1", "b": "2"}

    @pytest.mark.parametrize(
        ("text", "why"),
        [("a==1\n", "unhashed"), (f"a>=1 {H}\n", "not an exact pin"), (f"a {H}\n", "exact")],
    )
    def test_refuses_what_is_not_a_lock(self, tmp_path: Path, text: str, why: str) -> None:
        with pytest.raises(LockError, match=why):
            parse_lock(lock_file(tmp_path, text))

    def test_refuses_two_pins_for_one_environment(self, tmp_path: Path) -> None:
        with pytest.raises(LockError, match="pinned twice"):
            parse_lock(lock_file(tmp_path, f"a==1 {H}\na==2 {H}\n"))


# ── The interpreter, to the patch release ────────────────────────────────────


class TestThePythonRelease:
    def test_read_from_the_header_and_through_an_include(self, tmp_path: Path) -> None:
        lock_file(tmp_path, f"#   Python        3.13.15\na==1 {H}\n")
        dev = lock_file(tmp_path, f"-r requirements.lock\nb==2 {H}\n", "dev.lock")
        assert lock_python(dev) == "3.13.15"

    def test_a_lock_that_states_no_release_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(LockError, match="states no Python release"):
            lock_python(lock_file(tmp_path, f"a==1 {H}\n"))

    def test_another_patch_release_is_a_difference(self) -> None:
        """The 2026-10-02 rebuild ran 3.13.16 where the measured image ran 3.13.15."""
        assert python_problems("3.13.15", "3.13.15") == []
        assert python_problems("3.13.15", "3.13.16") == [
            "python: running 3.13.16, the lock says 3.13.15"
        ]

    def test_the_lock_the_image_base_and_every_ci_job_agree(self) -> None:
        wanted = lock_python(LOCK)
        assert wanted == "3.13.15"
        dockerfile = (REPO / "infra" / "Dockerfile").read_text(encoding="utf-8")
        assert f"ARG PYTHON_IMAGE=python:{wanted}-slim@sha256:" in dockerfile
        workflow = yaml.safe_load(
            (REPO / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
        )
        for job_id, job in workflow["jobs"].items():
            versions = [
                str(step["with"]["python-version"])
                for step in job["steps"]
                if str(step.get("uses", "")).startswith("actions/setup-python")
            ]
            assert versions == [wanted], (job_id, versions)


# ── Comparing a lock with an installed set ───────────────────────────────────


class TestCompare:
    LOCK: ClassVar[dict[str, str]] = {"langfuse": "4.15.6", "torch": "2.14.0+cpu"}

    def test_identical_sets_pass(self) -> None:
        assert compare(self.LOCK, {"langfuse": ["4.15.6"], "torch": ["2.14.0+cpu"]}) == []

    @pytest.mark.parametrize(
        ("have", "expected"),
        [
            # The 2026-10-02 rebuild, in one line.
            ({"langfuse": ["4.16.0"], "torch": ["2.14.0+cpu"]}, "version: langfuse is 4.16.0"),
            ({"langfuse": ["4.15.6"], "torch": ["2.14.0"]}, "version: torch is 2.14.0,"),
            ({"torch": ["2.14.0+cpu"]}, "missing: langfuse==4.15.6"),
            (
                {"langfuse": ["4.15.6"], "torch": ["2.14.0+cpu"], "httpx2": ["2.13.1"]},
                "not in the lock: httpx2==2.13.1",
            ),
            (
                {"langfuse": ["4.15.6", "4.16.0"], "torch": ["2.14.0+cpu"]},
                "installed twice: langfuse",
            ),
        ],
    )
    def test_every_kind_of_difference_is_named(
        self, have: dict[str, list[str]], expected: str
    ) -> None:
        problems = compare(self.LOCK, have)
        assert len(problems) == 1 and problems[0].startswith(expected), problems


# ── The check against this interpreter's real installed set ──────────────────


def run_check(lock: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(REPO / "scripts" / "lock_check.py"), "--lock", str(lock)],
        capture_output=True,
        text=True,
    )


class TestTheCheckFiresOnThisInterpreter:
    """No fakes: the script, run as CI and the image build run it, on the installed set."""

    def test_this_environment_equals_the_dev_lock(self) -> None:
        """Fails on a venv built on any Python but the lock's release: rebuild it on that one
        (`rm -rf .venv && make install PYTHON=python3.13.15`)."""
        result = run_check(DEV_LOCK)
        assert result.returncode == 0, result.stdout + result.stderr

    def copies(self, tmp_path: Path) -> tuple[Path, Path]:
        runtime = lock_file(tmp_path, LOCK.read_text(encoding="utf-8"))
        dev = lock_file(tmp_path, DEV_LOCK.read_text(encoding="utf-8"), DEV_LOCK.name)
        return runtime, dev

    def test_a_moved_version_fails_and_is_named(self, tmp_path: Path) -> None:
        runtime, dev = self.copies(tmp_path)
        text = runtime.read_text(encoding="utf-8")
        assert "\nlangfuse==4.15.6 " in text
        runtime.write_text(text.replace("\nlangfuse==4.15.6 ", "\nlangfuse==4.16.0 "))
        result = run_check(dev)
        assert result.returncode == 1
        assert "version: langfuse is 4.15.6, the lock says 4.16.0" in result.stdout

    def test_another_python_release_fails_and_is_named(self, tmp_path: Path) -> None:
        runtime, dev = self.copies(tmp_path)
        text = runtime.read_text(encoding="utf-8")
        assert "#   Python        3.13.15\n" in text
        runtime.write_text(
            text.replace("#   Python        3.13.15\n", "#   Python        3.13.99\n")
        )
        result = run_check(dev)
        assert result.returncode == 1
        assert "the lock says 3.13.99" in result.stdout

    def test_a_package_the_lock_does_not_name_fails(self, tmp_path: Path) -> None:
        runtime, dev = self.copies(tmp_path)
        text = runtime.read_text(encoding="utf-8")
        dropped = re.sub(r"\nlangfuse==4\.15\.6 \\\n(    --hash=\S+( \\)?\n)+", "\n", text)
        assert dropped != text
        runtime.write_text(dropped)
        result = run_check(dev)
        assert result.returncode == 1
        assert "not in the lock: langfuse==4.15.6" in result.stdout


# ── The committed locks ──────────────────────────────────────────────────────


def pyproject() -> dict[str, list[str]]:
    project = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    return {"runtime": project["dependencies"], "dev": project["optional-dependencies"]["dev"]}


class TestTheCommittedLocks:
    def test_the_runtime_lock_is_read_off_the_load_checked_image(self) -> None:
        header = LOCK.read_text(encoding="utf-8").split("\n\n", 1)[0]
        assert "godvillain-scholium:cpu-caefad0" in header
        assert "sha256:5f5b8c527cb56580b303b6770a7267c8590c06c02e7ec4aaf3e22623e857dbed" in header
        assert "Python        3.13.15" in header

    def test_linux_and_macos_differ_only_in_torch_s_build(self) -> None:
        linux, darwin = parse_lock(LOCK, LINUX), parse_lock(LOCK, DARWIN)
        assert linux.keys() == darwin.keys()
        assert {n for n in linux if linux[n] != darwin[n]} == {"torch"}
        assert (linux["torch"], darwin["torch"]) == ("2.14.0+cpu", "2.14.0")

    @pytest.mark.parametrize("env", [LINUX, DARWIN], ids=["linux", "darwin"])
    def test_every_declared_dependency_is_pinned_and_in_range(self, env: dict[str, str]) -> None:
        """A dependency `pyproject.toml` declares and the lock lacks would install nowhere,
        since the project is installed with --no-deps; one out of range fails `pip check`."""
        runtime = parse_lock(LOCK, env)
        dev = parse_lock(DEV_LOCK, env)
        for spec, pins in [(s, runtime) for s in pyproject()["runtime"]] + [
            (s, dev) for s in pyproject()["dev"]
        ]:
            req = Requirement(spec)
            name = canonicalize_name(req.name)
            assert name in pins, f"{name} is declared but not locked"
            assert req.specifier.contains(pins[name], prereleases=True), (name, pins[name])

    def test_the_dev_lock_adds_tools_and_repins_nothing(self) -> None:
        text = DEV_LOCK.read_text(encoding="utf-8")
        assert "\n-r requirements.lock\n" in f"\n{text}"
        runtime = parse_lock(LOCK, LINUX)
        own = [ln.split("==")[0] for ln in text.splitlines() if re.match(r"^[a-z0-9-]+==", ln)]
        assert own and not set(own) & runtime.keys()

    def test_the_dev_lock_is_generated_from_requirements_dev_in(self) -> None:
        wanted = {
            canonicalize_name(Requirement(ln).name): str(Requirement(ln).specifier)[2:]
            for ln in DEV_IN.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.startswith("#")
        }
        runtime = parse_lock(LOCK, LINUX)
        tools = {n: v for n, v in parse_lock(DEV_LOCK, LINUX).items() if n not in runtime}
        assert tools == wanted


# ── Every install goes through the lock ──────────────────────────────────────


def pip_installs(text: str) -> list[str]:
    return [ln.strip() for ln in text.splitlines() if re.search(r"\bpip install\b", ln)]


class TestEveryInstallUsesTheLock:
    def dockerfile(self) -> str:
        return (REPO / "infra" / "Dockerfile").read_text(encoding="utf-8")

    def test_the_base_image_is_pinned_by_digest_in_both_stages(self) -> None:
        text = self.dockerfile()
        (arg,) = re.findall(r"^ARG PYTHON_IMAGE=(\S+)$", text, re.M)
        assert re.fullmatch(r"python:3\.13\.15-slim@sha256:[0-9a-f]{64}", arg)
        froms = re.findall(r"^FROM (\S+)", text, re.M)
        assert froms.count("${PYTHON_IMAGE}") == 2 and all("python:" not in f for f in froms)

    def test_the_image_installs_the_lock_and_then_must_equal_it(self) -> None:
        text = self.dockerfile()
        installs = pip_installs(text)
        assert installs[0] == "RUN pip install --require-hashes -r requirements.lock"
        assert all("--require-hashes" in i or "--no-deps" in i for i in installs), installs
        # The check comes after the last install, so nothing installed later escapes it.
        last_install = max(text.rfind(i) for i in installs)
        check = text.find("RUN python lock_check.py --lock requirements.lock")
        assert check > last_install
        assert text.count("pip install") == len(installs)

    def test_the_lock_and_its_check_are_deployed(self) -> None:
        from scripts.deploy_space import deployed_paths

        assert {"requirements.lock", "scripts/lock_check.py"} <= set(deployed_paths(REPO))

    def test_every_ci_job_installs_the_lock_and_checks_it(self) -> None:
        workflow = yaml.safe_load(
            (REPO / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
        )
        for job_id, job in workflow["jobs"].items():
            runs = [s.get("run", "") for s in job["steps"]]
            joined = "\n".join(runs)
            installs = pip_installs(joined)
            assert installs, f"{job_id} installs nothing"
            assert all("--require-hashes" in i or "--no-deps" in i for i in installs), installs
            lock = "requirements.lock" if job_id == "clean-install" else "requirements-dev.lock"
            assert f"pip install --require-hashes -r {lock}" in joined, job_id
            check = f"python scripts/lock_check.py --lock {lock}"
            assert check in joined, f"{job_id} never checks its installed set"
            assert joined.index(check) > max(joined.index(i) for i in installs), job_id

    def test_make_install_uses_the_lock_and_checks_it(self) -> None:
        makefile = (REPO / "Makefile").read_text(encoding="utf-8")
        recipe = makefile.split("\ninstall:", 1)[1].split("\n\n", 1)[0]
        assert "--require-hashes -r requirements-dev.lock" in recipe
        assert all("--require-hashes" in i or "--no-deps" in i for i in pip_installs(recipe))
        assert "scripts/lock_check.py --lock requirements-dev.lock" in recipe
