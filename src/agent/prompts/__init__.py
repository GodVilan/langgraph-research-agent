"""Versioned prompt loading.

Prompts live in ``*.vN.md`` files beside this module, never as inline strings.

``RequestOptions.prompt_version`` names a *prompt set*, not one file. Prompts do not all
change together — Phase 2 rewrote ``generate`` to carry the injection framing while
``scope``, ``plan``, and ``critique`` were untouched — so requesting set ``v2`` resolves
each prompt to the highest version it actually has at or below ``v2``.

That resolution is explicit rather than silent: ``resolve_version`` is a public function,
nodes record what they resolved to, and requesting a version below everything available is
an error rather than a quiet substitution. Copying four files forward on every prompt change
would invite exactly the drift this repo is trying to avoid.
"""

from __future__ import annotations

import functools
import re
from pathlib import Path

PROMPT_DIR = Path(__file__).parent
_VERSION_RE = re.compile(r"^v(\d+)$")


def _version_number(version: str) -> int:
    match = _VERSION_RE.match(version)
    if match is None:
        raise ValueError(f"prompt version must look like 'v1', got {version!r}")
    return int(match.group(1))


@functools.lru_cache(maxsize=64)
def available_versions(name: str) -> tuple[str, ...]:
    """Versions on disk for one prompt, ascending."""
    found = []
    for path in PROMPT_DIR.glob(f"{name}.v*.md"):
        version = path.name[len(name) + 1 : -len(".md")]
        if _VERSION_RE.match(version):
            found.append(version)
    return tuple(sorted(found, key=_version_number))


@functools.lru_cache(maxsize=64)
def resolve_version(name: str, requested: str = "v1") -> str:
    """Highest version of ``name`` at or below ``requested``.

    Raises if the prompt has no version that old, rather than silently serving a newer one
    than the caller asked for.
    """
    versions = available_versions(name)
    if not versions:
        raise FileNotFoundError(
            f"No prompt files for {name!r} in {PROMPT_DIR}. "
            f"Available: {sorted(p.name for p in PROMPT_DIR.glob('*.md'))}"
        )
    ceiling = _version_number(requested)
    eligible = [v for v in versions if _version_number(v) <= ceiling]
    if not eligible:
        raise FileNotFoundError(
            f"Prompt {name!r} has no version at or below {requested!r}; it starts at "
            f"{versions[0]!r}. Requesting an older prompt set than a prompt has ever had is "
            f"a configuration error, not something to paper over."
        )
    return eligible[-1]


@functools.lru_cache(maxsize=64)
def load_prompt(name: str, version: str = "v1") -> str:
    """Load the prompt for ``name`` under prompt set ``version``."""
    resolved = resolve_version(name, version)
    return (PROMPT_DIR / f"{name}.{resolved}.md").read_text(encoding="utf-8").strip()


def prompt_manifest(version: str = "v1") -> dict[str, str]:
    """What each prompt resolves to under a given set. Recorded on the trace in Phase 3."""
    names = {path.name.rsplit(".", 2)[0] for path in PROMPT_DIR.glob("*.v*.md")}
    return {name: resolve_version(name, version) for name in sorted(names)}


def available_prompts() -> dict[str, list[str]]:
    """Map prompt name -> versions on disk."""
    names = {path.name.rsplit(".", 2)[0] for path in PROMPT_DIR.glob("*.v*.md")}
    return {name: list(available_versions(name)) for name in sorted(names)}
