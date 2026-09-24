"""Copy the deployed service's secrets from a deploy env file into a Hugging Face Space.

    make space-secrets SPACE=<owner>/<name> [ENV_FILE=.env.deploy]

Reads ``.env.deploy`` by default, not ``.env``: the local ``.env`` keeps the localhost Langfuse
fixture keys, the deploy file holds the Cloud ones. The file must be gitignored — asserted with
``git check-ignore`` before it is read.

Meant to be run by the Space's owner, not by automation: it moves real credentials to a
third party. Prints names, never values. Refuses the seeded Langfuse fixture pair outright
(tests/test_deploy_guards.py), and refuses a Langfuse host that is not Langfuse Cloud-shaped
HTTPS, since a deployed instance has no localhost Langfuse to reach.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# Secret on the Space: private, not copied when the Space is duplicated.
SECRETS = (
    "GOOGLE_API_KEY",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
    "UPSTASH_REDIS_REST_URL",
    "UPSTASH_REDIS_REST_TOKEN",
    "LOADCHECK_TOKEN",
)
# Variables: public configuration, visible on the Space's settings page.
VARIABLES = {
    "LANGFUSE_HOST": "https://cloud.langfuse.com",
    "API__LEDGER": "upstash",
}


def assert_gitignored(path: Path) -> None:
    """Refuse to read a credentials file git would commit.

    ``git check-ignore`` exits 0 only for an untracked path an ignore rule matches; a tracked
    file, or one no rule covers, exits non-zero — both mean the real keys could be committed.
    """
    import subprocess

    result = subprocess.run(
        ["git", "check-ignore", "-q", str(path)], cwd=REPO, capture_output=True, check=False
    )
    if result.returncode != 0:
        raise SystemExit(
            f"refusing to read {path}: it is not gitignored (or is tracked). It holds deploy "
            f"credentials; add it to .gitignore and make sure it was never committed."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--space", required=True)
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env.deploy"),
        help="the deploy credentials file (default .env.deploy); local .env keeps the fixtures",
    )
    args = parser.parse_args()

    from dotenv import dotenv_values
    from huggingface_hub import HfApi

    from src.observability.config import SEEDED_KEYS

    env_file = args.env_file if args.env_file.is_absolute() else REPO / args.env_file
    assert_gitignored(env_file)
    if not env_file.is_file():
        raise SystemExit(f"{env_file} does not exist")
    env = {k: v for k, v in dotenv_values(env_file).items() if v}
    if any(env.get(k) in SEEDED_KEYS for k in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")):
        raise SystemExit(
            f"LANGFUSE_* in {env_file.name} are the seeded localhost fixture keys. Put the "
            f"Langfuse Cloud project's key pair in {env_file.name} (docs/SERVING.md)."
        )
    host = env.get("LANGFUSE_HOST", VARIABLES["LANGFUSE_HOST"])
    if not host.startswith("https://"):
        raise SystemExit(f"LANGFUSE_HOST={host!r}: a deployed instance needs an https host")

    api = HfApi()
    for name in SECRETS:
        if name in env:
            api.add_space_secret(args.space, name, env[name])
            print(f"secret   {name}")
        else:
            print(f"skipped  {name} (not in {env_file.name})")
    for name, default in VARIABLES.items():
        value = host if name == "LANGFUSE_HOST" else env.get(name, default)
        api.add_space_variable(args.space, name, value)
        print(f"variable {name}={value}")


if __name__ == "__main__":
    main()
