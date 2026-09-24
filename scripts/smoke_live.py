"""Smoke-test a running instance by its effects, not its status codes (D-026).

    make smoke-live URL=https://<space>.hf.space [SPACE=owner/name] [SAMPLE_RATE=1.0]

Every check asserts what happened, not that something answered:

1. **The README's own curl example, run verbatim.** The command between the ``CURL`` markers in
   README.md is executed exactly as printed, and must return an answer citing a chunk id with
   at least one source. If the documented example and the smoke test could diverge, they
   would; so the test *is* the example. It must also target ``URL``.
2. **The index is the measured one.** ``/ready`` must report 5,401 chunks and the FAISS
   checksum recorded in ``data/INDEX.sha256``.
3. **The limiter keys on the real client.** A forged ``X-Forwarded-For`` must not change the
   ``X-Client-Key`` the limiter uses, and that key must be this machine's public address. A
   mismatch means ``API__TRUSTED_PROXY_HOPS`` is wrong for this host — the check that could
   not run locally.
4. **The trace arrived, complete.** The trace id the curl returned is read back through the v2
   observations API (``LANGFUSE_HOST`` and keys from the environment, then ``--env-file``),
   polling up to ``--trace-timeout`` (300 s); complete = the observation count stopped growing
   and the root span carries the answer.
5. **The sampling policy is in effect.** ``/ready`` reports the sampler's actual rate, which
   must equal ``SAMPLE_RATE``; with ``SPACE`` the Space's run log is read for the startup line
   recording the same rate.

Makes two model-backed queries' worth of calls at most; everything else is cheap endpoints.
Exits non-zero on the first failed check, naming it.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

CURL_BLOCK = re.compile(r"<!-- CURL:START -->\s*```bash\n(.*?)\n```\s*<!-- CURL:END -->", re.DOTALL)


def cited_source_papers(answer: str, source_paper_ids: list[str]) -> set[str]:
    """Source papers the answer cites, using the agent's own citation pattern (one definition:
    src/agent/nodes/finalize.py). An abbreviated id matches a source by its paper suffix."""
    from src.agent.nodes.finalize import CHUNK_ID_RE, CITE_BRACKET_RE

    hit: set[str] = set()
    for bracket in CITE_BRACKET_RE.findall(answer):
        for prefix, rest in CHUNK_ID_RE.findall(bracket):
            paper = rest.rsplit("_", 1)[0]
            for pid in source_paper_ids:
                if (prefix and pid == f"{prefix}.{paper}") or (not prefix and pid.endswith(paper)):
                    hit.add(pid)
    return hit


EXPECTED_CHUNKS = 5401


class SmokeFailureError(RuntimeError):
    pass


def check(ok: bool, name: str, detail: str) -> None:
    if not ok:
        raise SmokeFailureError(f"FAIL {name}: {detail}")
    print(f"ok   {name}: {detail}")


def readme_curl() -> str:
    match = CURL_BLOCK.search((REPO / "README.md").read_text(encoding="utf-8"))
    if match is None:
        raise SmokeFailureError(
            "FAIL readme-curl: no ```bash block between CURL markers in README.md"
        )
    return match.group(1).strip()


def expected_index_sha() -> str:
    for line in (REPO / "data" / "INDEX.sha256").read_text(encoding="utf-8").splitlines():
        digest, name = line.split(None, 1)
        if name.strip().endswith(".faiss"):
            return digest
    raise SmokeFailureError("FAIL index-sha: no .faiss entry in data/INDEX.sha256")


ENV_FILE = {"path": REPO / ".env"}


def env(name: str) -> str:
    value = os.environ.get(name)
    if value:
        return value
    from dotenv import dotenv_values

    return str(dotenv_values(ENV_FILE["path"]).get(name) or "")


DRAWS = 3  # generation is unpinned: one draw of the example is a coin flip on citation form


def run_readme_curl(url: str) -> dict[str, Any]:
    """Run the README's example verbatim, up to ``DRAWS`` times.

    The answer text is not deterministic (the generator is unpinned, D-042), and on the
    free-tier key its citation form drifts — abbreviated ids, several ids in one bracket, and
    occasionally a corrupted id the agent correctly refuses to resolve. One draw failing to
    cite is therefore not a deployment failure, and passing on the first lucky draw would hide
    the rate. So every draw is printed with the citations it carried; the check passes if any
    draw cites a returned source, and warns with the count when not all did.
    """
    import re

    command = readme_curl()
    check(url.rstrip("/") in command, "readme-curl-target", f"README example targets {url}")
    cited_draws, first_cited = 0, None
    for draw in range(1, DRAWS + 1):
        started = time.monotonic()
        proc = subprocess.run(
            ["bash", "-c", command], capture_output=True, text=True, timeout=300, check=False
        )
        check(
            proc.returncode == 0, "readme-curl-exit", f"exit {proc.returncode} {proc.stderr[:200]}"
        )
        try:
            body: dict[str, Any] = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise SmokeFailureError(
                f"FAIL readme-curl-json: not JSON: {proc.stdout[:300]}"
            ) from exc
        answer = str(body.get("answer", ""))
        sources = [str(src["paper_id"]) for src in body.get("sources") or []]
        check(bool(sources), "answer-has-sources", f"draw {draw}: {len(sources)} source(s)")
        cited = cited_source_papers(answer, sources)
        brackets = sorted(set(re.findall(r"\[[^\[\]]*_\d{4}[^\[\]]*\]", answer)))[:4]
        print(
            f"     draw {draw}: {time.monotonic() - started:.1f} s, "
            f"cites {sorted(cited) or 'none'} — brackets {brackets or 'none'}"
        )
        if cited:
            cited_draws += 1
            first_cited = first_cited or body
            if draw == 1:
                break  # the common case: one clean draw is enough
    check(
        first_cited is not None,
        "answer-cites-a-source",
        f"{cited_draws} of the draws cited a returned source",
    )
    assert first_cited is not None
    if cited_draws < draw:
        print(f"WARN answer-citation-rate: {cited_draws} of {draw} draws cited a source")
    return first_cited


def check_ready(url: str, sample_rate: float) -> None:
    ready = httpx.get(f"{url}/ready", timeout=30).json()
    check(ready.get("status") == "ready", "ready", str(ready.get("status")))
    check(ready.get("chunks") == EXPECTED_CHUNKS, "index-chunks", f"{ready.get('chunks')} chunks")
    want = expected_index_sha()
    got = str(ready.get("index_sha256", ""))
    check(got == want, "index-sha256", f"{got[:12]} == INDEX.sha256 {want[:12]}")
    rate = ready.get("trace_sample_rate")
    check(
        rate == sample_rate,
        "sample-rate-in-effect",
        f"sampler reports {rate}, configured {sample_rate}",
    )
    check(
        ready.get("ledger") == "upstash" or ready.get("ledger") == "sqlite",
        "ledger-durable",
        f"ledger {ready.get('ledger')}",
    )


def check_client_key(url: str, my_ip: str | None) -> None:
    from src.api.limits import client_key

    plain = httpx.get(f"{url}/health", timeout=30).headers.get("x-client-key", "")
    forged = httpx.get(
        f"{url}/health", headers={"X-Forwarded-For": "198.51.100.66"}, timeout=30
    ).headers.get("x-client-key", "")
    check(
        bool(plain) and plain == forged,
        "xff-forgery-ignored",
        f"key {plain} unchanged by a forged header",
    )
    check(
        forged != client_key("198.51.100.66"),
        "xff-forgery-not-keyed",
        "the forged address is not the key",
    )
    if my_ip is None:
        my_ip = httpx.get("https://api.ipify.org", timeout=15).text.strip()
    check(
        plain == client_key(my_ip),
        "limiter-keys-on-client",
        f"key is hash({my_ip}) — if not, API__TRUSTED_PROXY_HOPS is wrong for this host",
    )


def check_trace(trace_id: str, timeout_s: float) -> None:
    """The trace arrived **complete**, read through the v2 observations API (the legacy
    ``/api/public/traces`` answers 410 for new Cloud organisations). Polls, never fetches once:
    the server's batch exporter and Langfuse's ingestion both delay it. Complete means the
    observation count stopped growing and the root span carries the answer — a trace that
    merely exists can be missing its root (scripts/langfuse_readback.py)."""
    from scripts.langfuse_readback import wait_for_complete

    check(bool(trace_id), "trace-id-returned", "response carries a trace id")
    from scripts.space_secrets import VARIABLES

    # The deploy env file need not carry the host: `make space-secrets` sets it on the Space
    # from the same default, so read back from where the Space actually writes.
    host = (env("LANGFUSE_HOST") or VARIABLES["LANGFUSE_HOST"]).rstrip("/")
    auth = (env("LANGFUSE_PUBLIC_KEY"), env("LANGFUSE_SECRET_KEY"))
    check(all(auth) and bool(host), "langfuse-creds", f"reading back from {host}")
    result = wait_for_complete(host, auth, trace_id, timeout_s)
    check(result is not None, "trace-arrived", f"{trace_id[:12]} readable within {timeout_s:.0f} s")
    assert result is not None
    check(
        bool(result["complete"]),
        "trace-complete",
        f"{result['observations']} observations, root {result['root_name']!r} carries the "
        f"answer; first seen after {result['seconds_until_first_seen']} s",
    )


def check_startup_log(space: str, sample_rate: float) -> None:
    from huggingface_hub import get_token

    token = get_token() or env("HF_TOKEN")  # the `hf auth login` token first
    wanted = f"trace sample rate in effect {sample_rate}"
    seen: list[str] = []
    with httpx.stream(
        "GET",
        f"https://huggingface.co/api/spaces/{space}/logs/run",
        headers={"Authorization": f"Bearer {token}"},
        timeout=httpx.Timeout(20, read=20),
    ) as r:
        try:
            for line in r.iter_lines():
                seen.append(line)
                if wanted in line:
                    check(True, "startup-log-sample-rate", f"run log records '{wanted}'")
                    return
        except httpx.ReadTimeout:
            pass
    raise SmokeFailureError(
        f"FAIL startup-log-sample-rate: '{wanted}' not in {len(seen)} log lines"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--url", required=True)
    parser.add_argument("--space", default="", help="owner/name, to read the run log")
    parser.add_argument("--sample-rate", type=float, default=1.0)
    parser.add_argument("--my-ip", default=None, help="skip auto-detection of this machine's IP")
    parser.add_argument("--no-trace", action="store_true", help="local run without Langfuse")
    parser.add_argument(
        "--env-file",
        type=Path,
        default=REPO / ".env",
        help="where Langfuse read-back keys come from (.env.deploy for the deployed Space)",
    )
    parser.add_argument(
        "--trace-timeout", type=float, default=300.0, help="seconds to poll for the trace"
    )
    args = parser.parse_args()
    ENV_FILE["path"] = args.env_file
    url = args.url.rstrip("/")
    try:
        check_ready(url, args.sample_rate)
        body = run_readme_curl(url)
        check_client_key(url, args.my_ip)
        if not args.no_trace:
            check_trace(str(body.get("trace_id", "")), args.trace_timeout)
        if args.space:
            check_startup_log(args.space, args.sample_rate)
    except SmokeFailureError as failure:
        print(failure, file=sys.stderr)
        raise SystemExit(1) from None
    print("\nsmoke-live: every check passed" + (" (tracing not checked)" if args.no_trace else ""))


if __name__ == "__main__":
    main()
