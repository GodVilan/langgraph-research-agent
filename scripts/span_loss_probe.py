"""Are spans still buffered when the server stops exported, or lost?

The server does not flush per request (D-041): spans wait in the batch exporter and ship on
its schedule, and the server flushes once at shutdown, bounded at 10 s. So a trace for a
request answered moments before a stop exists only if that shutdown flush actually runs.

    python scripts/span_loss_probe.py local            # uvicorn on :8765 + the local Langfuse
    python scripts/span_loss_probe.py local-kill       # the same, stopped with SIGKILL
    python scripts/span_loss_probe.py space OWNER/NAME # the deployed Space; stops it with pause

``local`` starts the API against the local Langfuse (the seeded keys are legal on localhost),
answers one query, sends SIGTERM immediately — uvicorn's graceful path, which runs the
lifespan shutdown — then checks whether the trace arrived. ``space`` does the same against the
Space, stopping it with the Hub's pause (what a sleep does to a container is the host's
business; pause is the nearest stop we can trigger and observe), and restarts it afterwards.

Records the outcome in ``evals/runs/span_loss_<mode>.json``. Costs one model-backed query.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

QUESTION = {"question": "What is LoRA?", "stream": False}


def env(name: str, env_file: Path = REPO / ".env.deploy") -> str:
    """Space mode only: the deployed Langfuse is Cloud, whose keys live in .env.deploy — the
    local .env holds the localhost fixture pair."""
    from dotenv import dotenv_values

    return os.environ.get(name) or str(dotenv_values(env_file).get(name) or "")


def trace_arrives(
    trace_id: str, host: str, auth: tuple[str, str], wait_s: float
) -> dict[str, Any] | None:
    """Completeness, not existence: v2 observations API, observation count stable and the root
    span carrying the answer (scripts/langfuse_readback.py)."""
    from scripts.langfuse_readback import wait_for_complete

    return wait_for_complete(host, auth, trace_id, wait_s)


def local(kill: bool = False, langfuse_host: str = "http://localhost:3000") -> dict[str, Any]:
    port = 8765
    child_env = {
        **os.environ,
        "LANGFUSE_HOST": langfuse_host,
        "LANGFUSE_PUBLIC_KEY": "pk-lf-1a1a1a1a-2b2b-4c4c-8d8d-3e3e3e3e3e3e",
        "LANGFUSE_SECRET_KEY": "sk-lf-4f4f4f4f-5a5a-4b6b-8c7c-6d6d6d6d6d6d",
        "LANGFUSE_ENVIRONMENT": "span-loss-probe",  # never `development`, which budget reads
        "CHECKPOINT_DB": "/tmp/span-loss-probe-threads.sqlite",
    }
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.api.main:app", "--port", str(port)],
        cwd=REPO,
        env=child_env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            try:
                if httpx.get(f"{url}/ready", timeout=5).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(1)
        body = httpx.post(f"{url}/query", json=QUESTION, timeout=180).json()
    finally:
        stopped_at = time.monotonic()
        server.send_signal(signal.SIGKILL if kill else signal.SIGTERM)
        server.wait(timeout=60)
    trace_id = str(body.get("trace_id", ""))
    arrived = trace_arrives(
        trace_id,
        langfuse_host,
        (child_env["LANGFUSE_PUBLIC_KEY"], child_env["LANGFUSE_SECRET_KEY"]),
        120,
    )
    return {
        "mode": "local-kill" if kill else "local",
        "stop": (
            "SIGKILL immediately after the response (no shutdown code runs)"
            if kill
            else "SIGTERM immediately after the response (uvicorn graceful shutdown)"
        ),
        "shutdown_s": round(time.monotonic() - stopped_at, 1),
        "trace_id": trace_id,
        "trace": arrived,
        "complete": bool(arrived and arrived["complete"]),
    }


def space(space_id: str) -> dict[str, Any]:
    from huggingface_hub import HfApi

    api = HfApi()  # the `hf auth login` token; write access to the Space is needed to pause it
    url = "https://" + space_id.replace("/", "-").replace("_", "-").lower() + ".hf.space"
    body = httpx.post(f"{url}/query", json=QUESTION, timeout=300).json()
    api.pause_space(space_id)
    trace_id = str(body.get("trace_id", ""))
    auth = (env("LANGFUSE_PUBLIC_KEY"), env("LANGFUSE_SECRET_KEY"))
    host = env("LANGFUSE_HOST") or "https://cloud.langfuse.com"  # the space-secrets default
    arrived = trace_arrives(trace_id, host.rstrip("/"), auth, 300)
    api.restart_space(space_id)
    return {
        "mode": "space",
        "space": space_id,
        "stop": "HfApi.pause_space immediately after the response",
        "trace_id": trace_id,
        "trace": arrived,
        "complete": bool(arrived and arrived["complete"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("mode", choices=["local", "local-kill", "space"])
    parser.add_argument("space_id", nargs="?", default="")
    parser.add_argument("--langfuse-host", default="http://localhost:3000")
    args = parser.parse_args()
    kill = args.mode == "local-kill"
    result = (
        space(args.space_id)
        if args.mode == "space"
        else local(kill=kill, langfuse_host=args.langfuse_host)
    )
    result["probed_utc"] = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
    out = REPO / "evals" / "runs" / f"span_loss_{args.mode}.json"
    out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
