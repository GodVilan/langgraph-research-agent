"""Fail unless the installed Python packages are exactly the lock (DECISIONS D-064).

    python scripts/lock_check.py --lock requirements.lock        # the image, clean-install
    python scripts/lock_check.py --lock requirements-dev.lock    # CI's jobs, the local venv

Until D-064 the image ran `pip install .` against open lower bounds, so every rebuild resolved
whatever PyPI held that day: the 2026-10-02 rebuild came up with 22 packages at newer versions,
2 new ones and a newer Python, and its traces lost their root. The image, CI and the local venv
had each drifted to a different set, and none of them could be rolled back, because no file
recorded what any of them had held. The lock now records the set; this script makes a
difference from it an error rather than a surprise.

The comparison is the whole installed set, both ways: a package the lock does not name is as
much a drift as a version that moved, because it changes what imports resolve to. Only `pip`
itself and this project's own distribution are exempt. Lines whose environment marker does not
match this interpreter are skipped, which is how one lock carries torch's Linux `+cpu` build
for the image and CI and its plain build for macOS.

The interpreter is part of the set. The lock's header states the exact Python release it was
read from (`#   Python        3.13.15`); a different patch release fails the check, as a moved
package does — the 2026-10-02 rebuild that lost its trace roots ran 3.13.16 where the measured
image ran 3.13.15. The image pins it by base digest, CI by `setup-python`; a local venv on another
release fails until it is rebuilt on that one.

Runs inside the image build, so it uses the standard library and `packaging` (which the lock
pins) and nothing else.
"""

from __future__ import annotations

import argparse
import platform
import re
import sys
from collections.abc import Iterable
from importlib import metadata
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

# The header line `make lock` writes: the image's interpreter, exact to the patch release.
PYTHON_LINE = re.compile(r"^#\s+Python\s+(3\.\d+\.\d+)\s*$", re.M)
# Not part of the set: the installer itself, and the distribution the lock is installed for.
EXEMPT = frozenset({"pip", "arxiv-agent-v3"})


class LockError(ValueError):
    """The lock file cannot be read as a lock."""


def _logical_lines(text: str) -> Iterable[str]:
    """Join backslash continuations and drop comments, as pip reads a requirements file."""
    pending = ""
    for raw in text.splitlines():
        line = raw.split(" #", 1)[0].rstrip() if not raw.lstrip().startswith("#") else ""
        if line.endswith("\\"):
            pending += line[:-1] + " "
            continue
        joined = (pending + line).strip()
        pending = ""
        if joined:
            yield joined
    if pending.strip():
        yield pending.strip()


def parse_lock(path: Path, environment: dict[str, str] | None = None) -> dict[str, str]:
    """Canonical name -> exact version, for every line that applies to this environment.

    Follows `-r` includes. Refuses anything that is not an exact, hashed pin: a lock with a
    range or a missing hash is not a lock.
    """
    pins: dict[str, str] = {}
    for line in _logical_lines(path.read_text(encoding="utf-8")):
        if line.startswith(("-r ", "--requirement ")):
            included = path.parent / line.split(maxsplit=1)[1]
            for name, version in parse_lock(included, environment).items():
                _add(pins, name, version, f"{included.name} (via {path.name})")
            continue
        if line.startswith("-"):  # an option line: --extra-index-url and the like
            continue
        spec, _, hashes = line.partition(" --hash=")
        if not hashes:
            raise LockError(f"{path.name}: unhashed requirement {spec!r}")
        req = Requirement(spec)
        if req.marker is not None and not req.marker.evaluate(environment):
            continue
        specifiers = list(req.specifier)
        if len(specifiers) != 1 or specifiers[0].operator != "==":
            raise LockError(f"{path.name}: {spec!r} is not an exact pin")
        _add(pins, canonicalize_name(req.name), specifiers[0].version, path.name)
    return pins


def _add(pins: dict[str, str], name: str, version: str, where: str) -> None:
    if name in pins and pins[name] != version:
        raise LockError(f"{where}: {name} pinned twice for this environment")
    pins[name] = version


def lock_python(path: Path) -> str:
    """The Python release the lock was read from, from its header; follows `-r` includes."""
    text = path.read_text(encoding="utf-8")
    found = PYTHON_LINE.search(text)
    if found:
        return found.group(1)
    for line in _logical_lines(text):
        if line.startswith(("-r ", "--requirement ")):
            return lock_python(path.parent / line.split(maxsplit=1)[1])
    raise LockError(f"{path.name}: states no Python release")


def python_problems(wanted: str, running: str) -> list[str]:
    """The interpreter, compared to the patch release: 3.13.16 is not 3.13.15."""
    return [] if running == wanted else [f"python: running {running}, the lock says {wanted}"]


def installed() -> dict[str, list[str]]:
    """Canonical name -> every version installed (more than one is itself a defect)."""
    found: dict[str, list[str]] = {}
    for dist in metadata.distributions():
        name = canonicalize_name(dist.metadata["Name"])
        if name not in EXEMPT:
            found.setdefault(name, []).append(dist.version)
    return found


def compare(lock: dict[str, str], have: dict[str, list[str]]) -> list[str]:
    """Every difference between the lock and the installed set; empty means identical."""
    problems: list[str] = []
    for name in sorted(lock.keys() | have.keys()):
        versions = have.get(name, [])
        if name not in lock:
            problems.append(f"not in the lock: {name}=={', '.join(versions)}")
        elif not versions:
            problems.append(f"missing: {name}=={lock[name]}")
        elif len(versions) > 1:
            problems.append(f"installed twice: {name} {sorted(versions)}")
        elif Version(versions[0]) != Version(lock[name]):
            problems.append(f"version: {name} is {versions[0]}, the lock says {lock[name]}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--lock", type=Path, required=True)
    args = parser.parse_args(argv)

    lock = parse_lock(args.lock)
    running = platform.python_version()
    problems = python_problems(lock_python(args.lock), running) + compare(lock, installed())
    if problems:
        print(f"lock-check: the installed set differs from {args.lock} ({len(problems)}):")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print(
        f"lock-check: Python {running} and {len(lock)} packages, identical to {args.lock} "
        f"({sys.platform})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
