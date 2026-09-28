"""Refuse `make langfuse-up` while the MinIO volume cannot be written by the pinned image.

The compose file runs Chainguard's MinIO as a non-root user (uid 65532, D-057). The existing
local volume was created by the old root-run `minio/minio` and is root-owned, mode 755: the new
image cannot write it. `docker compose up -d` would recreate the MinIO container on that volume
anyway, and the stack that holds the Phase 4 trace window would come up with a MinIO that cannot
store anything. So before `up`, this checks — through a **read-only** mount, changing nothing —
whether the image's user could write `/data`, `/data/langfuse` and `/data/.minio.sys`, and if it
could not, prints the migration plan from docs/BACKLOG.md instead of starting anything.

A volume that does not exist yet is a fresh stack, and passes: Compose creates it writable.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]  # no stubs; scripts/ is outside `make type`

REPO = Path(__file__).resolve().parent.parent
COMPOSE = REPO / "infra" / "docker-compose.langfuse.yml"
BACKLOG = REPO / "docs" / "BACKLOG.md"
MIGRATION_ROW = "| **Migrate the local Langfuse MinIO volume**"
PATHS = ("/data", "/data/langfuse", "/data/.minio.sys")


def docker(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["docker", *args], capture_output=True, text=True)


def compose_minio(compose: dict[str, Any]) -> tuple[str, str]:
    """The MinIO image, and the name of its data volume under the effective project."""
    project = os.environ.get("COMPOSE_PROJECT_NAME") or str(compose.get("name") or "")
    service = compose["services"]["minio"]
    mount = next(v for v in service["volumes"] if str(v).endswith(":/data"))
    return str(service["image"]), f"{project}_{str(mount).split(':', 1)[0]}"


def image_uid(image: str) -> int:
    """The numeric user the image runs as; 0 for root or unset."""
    out = docker("image", "inspect", image, "--format", "{{.Config.User}}")
    if out.returncode != 0:  # not pulled yet: pull it, as `up` would
        if docker("pull", "-q", image).returncode != 0:
            raise SystemExit(f"refusing: cannot pull {image} to check which user it runs as")
        out = docker("image", "inspect", image, "--format", "{{.Config.User}}")
    user = out.stdout.strip().split(":", 1)[0]
    return int(user) if user.isdigit() else 0


def listing(volume: str, image: str) -> list[tuple[str, int, int, str]]:
    """(mode, uid, gid, path) for each path that exists in the volume, read-only."""
    script = "; ".join(f"[ -e {p} ] && ls -lnd {p}" for p in PATHS) + "; true"
    out = docker(
        "run", "--rm", "-v", f"{volume}:/data:ro", "--entrypoint", "sh", image, "-c", script
    )
    if out.returncode != 0:
        raise SystemExit(f"refusing: could not inspect volume {volume}: {out.stderr.strip()[:200]}")
    rows = []
    for line in out.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 9:
            rows.append((parts[0], int(parts[2]), int(parts[3]), parts[-1]))
    return rows


def unwritable(uid: int, rows: list[tuple[str, int, int, str]]) -> list[str]:
    """The paths `uid` could not write, judged from the permission bits alone."""
    if uid == 0:
        return []
    blocked = []
    for mode, owner, _gid, path in rows:
        writable = (owner == uid and mode[2] == "w") or mode[8] == "w"
        if not writable:
            blocked.append(f"{path} ({mode}, owner uid {owner})")
    return blocked


def migration_steps() -> str:
    """The plan as BACKLOG states it — printed from there, so there is one copy of it."""
    for line in BACKLOG.read_text(encoding="utf-8").splitlines():
        if line.startswith(MIGRATION_ROW):
            cells = [c.strip() for c in line.strip("|").split("|")]
            plan = cells[1]
            start = plan.find("**Plan, in order:**")
            return plan[start:] if start >= 0 else plan
    return "see docs/BACKLOG.md, 'Migrate the local Langfuse MinIO volume'"


def check(compose_path: Path = COMPOSE) -> str:
    """Why `make langfuse-up` must not run, or "" if it may."""
    compose = yaml.safe_load(compose_path.read_text(encoding="utf-8"))
    image, volume = compose_minio(compose)
    if docker("volume", "inspect", volume).returncode != 0:
        return ""  # no volume yet: a fresh stack, created writable
    uid = image_uid(image)
    blocked = unwritable(uid, listing(volume, image))
    if not blocked:
        return ""
    return (
        f"the MinIO volume {volume} cannot be written by {image.split('@')[0]}, which runs as "
        f"uid {uid}: " + "; ".join(blocked) + ". `up` would recreate MinIO on it (D-057)."
    )


def main() -> int:
    why = check()
    if not why:
        return 0
    print(f"refusing `make langfuse-up`: {why}\n", file=sys.stderr)
    print("Migrate the volume first (docs/BACKLOG.md):\n", file=sys.stderr)
    print(migration_steps(), file=sys.stderr)
    print("\nThe running stack is unaffected; nothing was changed.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
