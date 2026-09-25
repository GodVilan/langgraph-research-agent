"""Deploy the service to a Hugging Face Docker Space.

    make deploy-space SPACE=<owner>/<name>            # assemble, check, upload, print the URL
    make deploy-space SPACE=<owner>/<name> DRY=1      # assemble and check only; uploads nothing

What it does, in order:

1. **Assembles a build context** from exactly the files ``.dockerignore`` admits — the same
   allow-list a local ``docker build`` uses, so the Space and a local image are built from
   the same inputs — plus ``infra/Dockerfile`` at the root and a generated Space card.
2. **Checks it** (``check_context``): no compose file, no ``.env``, no Langfuse provisioning
   block, and the corpus and index checksums verified before anything leaves the machine.
3. **Refuses to upload anything that is not a tagged commit** (``deploy_refusal``): every
   deployed path must be clean in git and ``HEAD`` must carry a ``deploy-*`` tag, so the Space
   always maps to a commit someone chose to ship. Two deploys ran from uncommitted files and
   were mapped to git only afterwards (D-049, D-054); this makes a third impossible by
   accident (D-055). ``--allow-dirty`` overrides it, warns, and says so in the Space's commit
   message and in ``infra/deploy_log.jsonl``.
4. **Uploads it** to a *public* Space. Publishing is outward-facing, so the upload refuses to
   run without ``--i-confirmed-public`` (the Makefile passes it only when ``DRY`` is unset).
5. **Records the deploy** in ``infra/deploy_log.jsonl``: time, Space commit, git ``HEAD``, its
   ``deploy-*`` tags, and any dirty paths an override let through.

What it deliberately does not do: set secrets. ``GOOGLE_API_KEY``, the Langfuse Cloud keys
and the ledger's credentials are entered by the owner in the Space's settings (or with
``make space-secrets``, which the owner runs), never by this script — docs/SERVING.md lists
each one. Hugging Face builds the image from the uploaded context on its own builders.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
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


def admitted_paths(repo: Path = REPO) -> list[str]:
    """The ``!`` entries of ``.dockerignore``: the only files a deploy may contain."""
    lines = (repo / ".dockerignore").read_text(encoding="utf-8").splitlines()
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


DEPLOY_LOG = REPO / "infra" / "deploy_log.jsonl"
# Deployed besides the allow-list: the Dockerfile and .dockerignore go to the Space root, and
# this script renders the Space card, so its own text is part of what ships.
ALSO_DEPLOYED = ("infra/Dockerfile", ".dockerignore", "scripts/deploy_space.py")


@dataclass
class Provenance:
    head: str
    deploy_tags: list[str]
    dirty: list[str] = field(default_factory=list)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=True
    ).stdout


def deployed_paths(repo: Path = REPO) -> list[str]:
    return [*admitted_paths(repo), *ALSO_DEPLOYED]


def provenance(repo: Path = REPO) -> Provenance:
    """What would ship: HEAD, the deploy tags on it, and every deployed path git sees as changed
    or untracked. Gitignored paths (the FAISS index) are not git's to vouch for; their
    checksums are committed in data/INDEX.sha256 and verified by ``check_context``."""
    head = _git(repo, "rev-parse", "HEAD").strip()
    tags = _git(repo, "tag", "--points-at", "HEAD", "--list", "deploy-*").split()
    status = _git(
        repo, "status", "--porcelain", "--untracked-files=all", "--", *deployed_paths(repo)
    )
    dirty = sorted(line[3:].strip() for line in status.splitlines() if line.strip())
    return Provenance(head=head, deploy_tags=tags, dirty=dirty)


def deploy_refusal(prov: Provenance) -> str:
    """Why this tree must not be deployed, or "" if it may be."""
    problems = []
    if prov.dirty:
        shown = ", ".join(prov.dirty[:8]) + (" …" if len(prov.dirty) > 8 else "")
        problems.append(f"deployed files differ from HEAD: {shown}")
    if not prov.deploy_tags:
        problems.append(f"HEAD {prov.head[:7]} carries no deploy-* tag")
    if not problems:
        return ""
    return (
        "; ".join(problems)
        + ". Commit and tag (`git tag deploy-YYYY-MM-DD`), then deploy from the tag (D-055)."
    )


def record_deploy(entry: dict[str, object], log: Path = DEPLOY_LOG) -> None:
    """Append one line and read it back — a deploy record that did not land is not a record
    (D-026)."""
    line = json.dumps(entry, sort_keys=True)
    with log.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    if log.read_text(encoding="utf-8").splitlines()[-1] != line:
        raise SystemExit(f"deploy record did not land in {log}")


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
        "--allow-dirty",
        action="store_true",
        help="deploy files that are not a tagged commit; warns and is recorded (D-055)",
    )
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

    prov = provenance()
    refusal = deploy_refusal(prov)
    workdir = Path(tempfile.mkdtemp(prefix="space-"))
    context = assemble_context(workdir / "context")
    (context / "README.md").write_text(
        SPACE_CARD.format(github=GITHUB, host=host), encoding="utf-8"
    )
    check_context(context)
    files = sorted(p.relative_to(context).as_posix() for p in context.rglob("*") if p.is_file())
    size = sum((context / f).stat().st_size for f in files)
    print(f"context: {len(files)} files, {size / 1e6:.1f} MB, checks passed ({context})")

    print(
        f"source: HEAD {prov.head[:7]}, tags {prov.deploy_tags or 'none'}, "
        f"dirty deployed paths {len(prov.dirty)}"
    )
    if args.dry_run:
        for f in files:
            print("  ", f)
        if refusal:
            print(f"an upload would be refused: {refusal}")
        return
    if refusal and not args.allow_dirty:
        raise SystemExit(f"refusing to deploy: {refusal}")
    if refusal:
        print(
            f"WARNING: --allow-dirty: deploying anyway. {refusal} This deploy maps to no "
            "commit; it is recorded as such.",
            file=sys.stderr,
        )
    if not args.i_confirmed_public:
        raise SystemExit("refusing to publish without --i-confirmed-public")
    source = (
        f"{prov.deploy_tags[0]} ({prov.head[:7]})"
        if not refusal
        else f"UNCOMMITTED (--allow-dirty) on {prov.head[:7]}"
    )

    from huggingface_hub import HfApi

    api = HfApi()  # HF_TOKEN from the environment; needs write access to `owner`
    api.create_repo(args.space, repo_type="space", space_sdk="docker", exist_ok=True)
    commit = api.upload_folder(
        folder_path=str(context),
        repo_id=args.space,
        repo_type="space",
        commit_message=f"deploy arXiv Agent v3 from {source}",
        delete_patterns=["*"],  # the Space mirrors the context exactly; stale files go
    )
    # Assert the effect, not the call (D-026): the commit must exist on the Space.
    head = api.space_info(args.space).sha
    if not head or commit.oid != head:
        raise SystemExit(f"upload returned {commit.oid} but the Space head is {head}")
    record_deploy(
        {
            "utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            "space": args.space,
            "space_commit": commit.oid,
            "git_head": prov.head,
            "deploy_tags": prov.deploy_tags,
            "allow_dirty": bool(refusal),
            "dirty_paths": prov.dirty,
        }
    )
    print(f"pushed {commit.oid[:8]} to https://huggingface.co/spaces/{args.space} from {source}")
    print(f"it builds on Hugging Face's builders; then: https://{host}/ready")


if __name__ == "__main__":
    main()
