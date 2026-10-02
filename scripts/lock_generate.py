"""Write requirements.lock from a built image's installed set, with hashes (DECISIONS D-064).

    make lock IMAGE=registry.hf.space/godvillain-scholium:cpu-caefad0

The runtime lock is not resolved from `pyproject.toml`; it is *read off an image that was
measured*. `cpu-caefad0` is the build of Space commit `caefad03`, which served the published load
check (2026-09-25) and whose traces were complete. Reading the set from it means the lock is a
record of a system that was observed working, not a fresh resolution that has never run.

Every file PyPI publishes for each pinned version is hashed, so one lock installs on Linux
(the image, CI) and macOS (the local venv) alike, and pip refuses any file whose hash it does
not list. torch is the exception the image forces: it is built from the CPU-only index as
`2.14.0+cpu`, which has no macOS build, so torch gets two lines with exclusive markers — the
`+cpu` wheels on Linux, PyPI's macOS wheels of the same release on Darwin.

The dev lock layers on top: `requirements-dev.in` lists the dev tools at exact versions (those
the suite was verified with), and this script hashes them into `requirements-dev.lock`, which
includes the runtime lock with `-r`. The runtime set is therefore one file, shared by all three
environments; only the tools the image never installs live in the second.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

REPO = Path(__file__).resolve().parent.parent
RUNTIME_LOCK = REPO / "requirements.lock"
DEV_IN = REPO / "requirements-dev.in"
DEV_LOCK = REPO / "requirements-dev.lock"
CPU_INDEX = "https://download.pytorch.org/whl/cpu"
PROJECT = "arxiv-agent-v3"
# Not recorded in a lock: the installer, and this project itself.
SKIP = {"pip", PROJECT}


def image_set(image: str) -> tuple[list[tuple[str, str]], str, str]:
    """(name, version) for every distribution in the image's venv, the image id, its Python."""

    def run(*cmd: str) -> str:
        return subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--platform",
                "linux/amd64",
                "--entrypoint",
                cmd[0],
                image,
                *cmd[1:],
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout

    listing = json.loads(run("/opt/venv/bin/pip", "list", "--format=json"))
    python = run("/opt/venv/bin/python", "-c", "import sys; print(sys.version.split()[0])").strip()
    image_id = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", image],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    pins = sorted(
        (canonicalize_name(p["name"]), p["version"])
        for p in listing
        if canonicalize_name(p["name"]) not in SKIP
    )
    return pins, image_id, python


def _get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as response:  # fixed hosts: PyPI, the torch index
        body: bytes = response.read()
        return body


def pypi_files(name: str, version: str) -> list[tuple[str, str]]:
    """(filename, sha256) for every file PyPI holds for this exact release."""
    data = json.loads(_get(f"https://pypi.org/pypi/{name}/{version}/json"))
    files = [(u["filename"], u["digests"]["sha256"]) for u in data["urls"]]
    if not files:
        raise SystemExit(f"PyPI lists no files for {name}=={version}")
    return files


def cpu_index_files(name: str, version: str) -> list[tuple[str, str]]:
    """(filename, sha256) for every file of `version` on the CPU-only torch index."""
    page = _get(f"{CPU_INDEX}/{name}/").decode()
    wanted = f"{name}-{version}-"
    files = [
        (urllib.parse.unquote(href.rsplit("/", 1)[-1]), digest)
        for href, digest in re.findall(r'href="([^"#]+)#sha256=([0-9a-f]{64})"', page)
    ]
    files = [(f, d) for f, d in files if f.startswith(wanted)]
    if not files:
        raise SystemExit(f"the CPU index lists no files for {name}=={version}")
    return files


def entry(spec: str, files: list[tuple[str, str]]) -> str:
    hashes = sorted({digest for _, digest in files})
    return " \\\n    ".join([spec, *(f"--hash=sha256:{h}" for h in hashes)])


def runtime_entries(pins: list[tuple[str, str]]) -> list[str]:
    out = []
    for name, version in pins:
        if "+" in version:  # a local build: only the CPU index has it (torch)
            public = version.split("+", 1)[0]
            out.append(
                entry(
                    f'{name}=={version} ; sys_platform == "linux"', cpu_index_files(name, version)
                )
            )
            mac = [(f, d) for f, d in pypi_files(name, public) if "macosx" in f]
            out.append(entry(f'{name}=={public} ; sys_platform == "darwin"', mac))
        else:
            out.append(entry(f"{name}=={version}", pypi_files(name, version)))
    return out


def write_runtime(image: str) -> None:
    pins, image_id, python = image_set(image)
    if not python.startswith("3.13."):
        raise SystemExit(f"{image} runs Python {python}; the project is 3.13")
    header = [
        "# requirements.lock — the runtime set, for the image, CI and the local venv (D-064).",
        "# GENERATED by `make lock` — do not edit by hand. Regenerate only from an image that",
        "# was measured, and say which in DECISIONS.",
        f"#   source image  {image}",
        f"#   image id      {image_id}",
        f"#   Python        {python}",
        f"#   generated     {dt.date.today().isoformat()}",
        f"#   packages      {len(pins)} (pip and {PROJECT} itself are not recorded)",
        "# torch is the CPU index's +cpu build on Linux and PyPI's build of the same release on",
        "# macOS, which has no +cpu wheel; the markers are exclusive.",
        f"--extra-index-url {CPU_INDEX}",
        "",
    ]
    RUNTIME_LOCK.write_text("\n".join(header + runtime_entries(pins)) + "\n", encoding="utf-8")
    print(f"wrote {RUNTIME_LOCK.name}: {len(pins)} packages from {image} ({image_id[:19]})")


def write_dev() -> None:
    specs = [
        ln.strip()
        for ln in DEV_IN.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.lstrip().startswith("#")
    ]
    body = []
    for spec in specs:
        req = Requirement(spec)
        (pin,) = list(req.specifier)
        if pin.operator != "==":
            raise SystemExit(f"{DEV_IN.name}: {spec!r} is not an exact pin")
        body.append(
            entry(
                f"{canonicalize_name(req.name)}=={pin.version}", pypi_files(req.name, pin.version)
            )
        )
    header = [
        "# requirements-dev.lock — CI's jobs and the local venv: the runtime lock plus dev tools.",
        f"# GENERATED by `make lock` from {DEV_IN.name} — do not edit by hand (D-064).",
        "-r requirements.lock",
        "",
    ]
    DEV_LOCK.write_text("\n".join(header + body) + "\n", encoding="utf-8")
    print(f"wrote {DEV_LOCK.name}: {len(body)} dev packages over the runtime lock")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--image", help="regenerate the runtime lock from this image")
    args = parser.parse_args()
    if args.image:
        write_runtime(args.image)
    write_dev()


if __name__ == "__main__":
    sys.exit(main())
