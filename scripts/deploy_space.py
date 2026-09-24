"""Deploy the service to a Hugging Face Docker Space.

    make deploy-space SPACE=<owner>/<name>            # assemble, check, upload, print the URL
    make deploy-space SPACE=<owner>/<name> DRY=1      # assemble and check only; uploads nothing

What it does, in order:

1. **Assembles a build context** from exactly the files ``.dockerignore`` admits — the same
   allow-list a local ``docker build`` uses, so the Space and a local image are built from
   the same inputs — plus ``infra/Dockerfile`` at the root and a generated Space card.
2. **Checks it** (``check_context``): no compose file, no ``.env``, no Langfuse provisioning
   block, and the corpus and index checksums verified before anything leaves the machine.
3. **Uploads it** to a *public* Space. Publishing is outward-facing, so the upload refuses to
   run without ``--i-confirmed-public`` (the Makefile passes it only when ``DRY`` is unset).

What it deliberately does not do: set secrets. ``GOOGLE_API_KEY``, the Langfuse Cloud keys
and the ledger's credentials are entered by the owner in the Space's settings (or with
``make space-secrets``, which the owner runs), never by this script — docs/SERVING.md lists
each one. Hugging Face builds the image from the uploaded context on its own builders.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

GITHUB = "https://github.com/GodVilan/langgraph-research-agent"

SPACE_CARD = """---
title: arXiv Agent v3
emoji: 📄
colorFrom: gray
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
short_description: LangGraph research agent over a fixed 150-paper arXiv corpus
---

# arXiv Agent v3

A graph-orchestrated research agent over a fixed corpus of 150 arXiv cs.LG papers, served as
an HTTP API. This Space runs the container built from [{github}]({github}); the README there
documents the endpoints, the limits, the measured numbers, and what the system cannot do.

```bash
curl -s -X POST https://{host}/query -H 'Content-Type: application/json' \\
  -d '{{"question": "What is LoRA?", "stream": false}}'
```

On free hardware this Space sleeps when idle; the first request after a sleep waits for the
container to start and the 1.3 GB embedding model to load.
"""

# Strings whose presence in a deploy context means the local Langfuse fixture is going with
# it: the compose file's name, and the headless-provisioning variables that create the
# fixture project and key pair.
FORBIDDEN_NAMES = ("docker-compose", "compose.yml", "compose.yaml", ".env")
FORBIDDEN_TEXT = ("LANGFUSE_INIT_",)


class ContextError(RuntimeError):
    """The assembled context must not be deployed."""


def admitted_paths() -> list[str]:
    """The ``!`` entries of ``.dockerignore``: the only files a deploy may contain."""
    lines = (REPO / ".dockerignore").read_text(encoding="utf-8").splitlines()
    return [ln.strip()[1:].rstrip("/") for ln in lines if ln.strip().startswith("!")]


def assemble_context(dest: Path) -> Path:
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for rel in admitted_paths():
        src = REPO / rel
        if not src.exists():
            raise ContextError(f"{rel} is admitted by .dockerignore but missing; run `make index`")
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        else:
            shutil.copy2(src, target)
    shutil.copy2(REPO / "infra" / "Dockerfile", dest / "Dockerfile")
    shutil.copy2(REPO / ".dockerignore", dest / ".dockerignore")
    return dest


def check_context(context: Path) -> None:
    """Refuse a context carrying the local fixture, or data that is not what Phase 4 measured."""
    for path in context.rglob("*"):
        rel = path.relative_to(context).as_posix()
        if any(bad in path.name for bad in FORBIDDEN_NAMES):
            raise ContextError(f"{rel}: a compose or env file must never be deployed")
        if path.is_file() and path.suffix in {".py", ".yml", ".yaml", ".toml", ".md", ".txt"}:
            text = path.read_text(encoding="utf-8", errors="replace")
            for bad in FORBIDDEN_TEXT:
                if bad in text:
                    raise ContextError(f"{rel}: contains {bad}… — the Langfuse provisioning block")

    for manifest in ("data/CORPUS.sha256", "data/INDEX.sha256"):
        for line in (context / manifest).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            expected, name = line.split(None, 1)
            digest = hashlib.sha256((context / name.strip()).read_bytes()).hexdigest()
            if digest != expected:
                raise ContextError(f"{name.strip()} does not match {manifest}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy to a Hugging Face Docker Space.")
    parser.add_argument("--space", required=True, help="owner/name")
    parser.add_argument("--dry-run", action="store_true", help="assemble and check only")
    parser.add_argument(
        "--i-confirmed-public",
        action="store_true",
        help="required to upload: the Space is public and serves on a real URL",
    )
    args = parser.parse_args()
    owner, _, name = args.space.partition("/")
    if not owner or not name:
        raise SystemExit("--space must be owner/name")
    host = f"{owner}-{name}".lower().replace("_", "-") + ".hf.space"

    workdir = Path(tempfile.mkdtemp(prefix="space-"))
    context = assemble_context(workdir / "context")
    (context / "README.md").write_text(
        SPACE_CARD.format(github=GITHUB, host=host), encoding="utf-8"
    )
    check_context(context)
    files = sorted(p.relative_to(context).as_posix() for p in context.rglob("*") if p.is_file())
    size = sum((context / f).stat().st_size for f in files)
    print(f"context: {len(files)} files, {size / 1e6:.1f} MB, checks passed ({context})")

    if args.dry_run:
        for f in files:
            print("  ", f)
        return
    if not args.i_confirmed_public:
        raise SystemExit("refusing to publish without --i-confirmed-public")

    from huggingface_hub import HfApi

    api = HfApi()  # HF_TOKEN from the environment; needs write access to `owner`
    api.create_repo(args.space, repo_type="space", space_sdk="docker", exist_ok=True)
    commit = api.upload_folder(
        folder_path=str(context),
        repo_id=args.space,
        repo_type="space",
        commit_message="deploy arXiv Agent v3",
        delete_patterns=["*"],  # the Space mirrors the context exactly; stale files go
    )
    # Assert the effect, not the call (D-026): the commit must exist on the Space.
    head = api.space_info(args.space).sha
    if not head or commit.oid != head:
        raise SystemExit(f"upload returned {commit.oid} but the Space head is {head}")
    print(f"pushed {commit.oid[:8]} to https://huggingface.co/spaces/{args.space}")
    print(f"it builds on Hugging Face's builders; then: https://{host}/ready")


if __name__ == "__main__":
    main()
