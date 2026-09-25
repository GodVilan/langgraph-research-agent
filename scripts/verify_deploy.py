"""Does the live Space serve exactly the files of a given git commit?

    make verify-deploy SPACE=godvillain/Scholium REV=d6b0369
    make verify-deploy SPACE=godvillain/Scholium REV=WORKTREE   # before the commit exists

Downloads the Space's tree at its current head and compares every file with ``REV``:

* repo files byte-for-byte (``Dockerfile`` against ``infra/Dockerfile``, which is what the deploy
  copies to the Space root);
* the gitignored FAISS index by sha256 against ``data/INDEX.sha256`` as committed at ``REV``;
* the Space card against what ``scripts/deploy_space.py`` *at REV* renders;
* ``.gitattributes`` is Hugging Face's own (LFS rules) and is listed, not compared.

Exits non-zero on any mismatch. Read-only against the Hub (uses the `hf auth login` token).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import subprocess
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


WORKTREE = "WORKTREE"


def git_blob(rev: str, path: str) -> bytes | None:
    """``path`` as committed at ``rev`` — or, with ``rev == WORKTREE``, as it is on disk now
    (for verifying a deploy made before the commit that will carry it exists)."""
    if rev == WORKTREE:
        f = REPO / path
        return f.read_bytes() if f.is_file() else None
    r = subprocess.run(["git", "show", f"{rev}:{path}"], cwd=REPO, capture_output=True, check=False)
    return r.stdout if r.returncode == 0 else None


def space_card_at(rev: str, host: str) -> str:
    src = git_blob(rev, "scripts/deploy_space.py")
    if src is None:
        raise SystemExit(f"scripts/deploy_space.py not in {rev}")
    with tempfile.NamedTemporaryFile("wb", suffix=".py", delete=False) as fh:
        fh.write(src)
    spec = importlib.util.spec_from_file_location("deploy_space_at_rev", fh.name)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return str(mod.SPACE_CARD.format(github=mod.GITHUB, host=host))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--space", required=True)
    parser.add_argument("--rev", required=True)
    args = parser.parse_args()

    from huggingface_hub import HfApi, snapshot_download

    head = HfApi().space_info(args.space).sha
    if head is None:
        raise SystemExit(f"{args.space}: the Hub reports no head commit")
    tree = Path(
        snapshot_download(
            args.space, repo_type="space", revision=head, local_dir=tempfile.mkdtemp()
        )
    )
    owner, name = args.space.split("/", 1)
    host = f"{owner}-{name}".lower().replace("_", "-") + ".hf.space"
    sums: dict[str, str] = {}
    for line in (git_blob(args.rev, "data/INDEX.sha256") or b"").decode().splitlines():
        digest, path = line.split(None, 1)
        sums[path.strip()] = digest

    identical: list[str] = []
    mismatched: list[str] = []
    listed: list[str] = []
    for f in sorted(p for p in tree.rglob("*") if p.is_file() and ".cache" not in p.parts):
        rel = f.relative_to(tree).as_posix()
        data = f.read_bytes()
        if rel == ".gitattributes":
            listed.append(f"{rel}: Hugging Face's own LFS rules, not a repo file")
        elif rel == "README.md":
            ok = data.decode() == space_card_at(args.rev, host)
            (identical if ok else mismatched).append(
                f"{rel}: Space card vs deploy_space.py@{args.rev}"
            )
        elif rel in sums:
            ok = hashlib.sha256(data).hexdigest() == sums[rel]
            (identical if ok else mismatched).append(
                f"{rel}: sha256 vs data/INDEX.sha256@{args.rev}"
            )
        else:
            src = "infra/Dockerfile" if rel == "Dockerfile" else rel
            blob = git_blob(args.rev, src)
            ok = blob == data
            (identical if ok else mismatched).append(
                f"{rel}: {'==' if ok else '!='} {src}@{args.rev}" + ("" if blob else " (absent)")
            )

    print(f"Space {args.space} @ {head[:8]}  vs  git {args.rev}")
    print(f"identical {len(identical)}, mismatched {len(mismatched)}, not compared {len(listed)}")
    for line in mismatched:
        print("  MISMATCH", line)
    for line in listed:
        print("  listed  ", line)
    if mismatched:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
