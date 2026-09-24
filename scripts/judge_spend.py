"""Total OpenAI judge spend, summed from the batches themselves.

Backs `make judge-spend`. Every other number in this project is regenerable by a command, and
the headline spend figure should not be the exception — it was previously a hand-tally of
figures `collect` printed to a terminal, which is exactly the "emit numbers, never type them"
rule (CLAUDE.md §3, pattern 6).

Source of truth: the batch receipts in `evals/runs/batch_*.json`, whose ids address the batch
objects and their output files at the provider. Token usage is read per response from each
output file and priced at the **batch** rates in docs/BUDGET.md (half of standard). A batch
whose output file has aged out of the provider's retention window is reported as unavailable
rather than silently omitted.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RUNS = Path("evals/runs")
BASE = "https://api.openai.com/v1"
# docs/BUDGET.md: gpt-5.6-luna standard $0.20 / $1.20 per 1M; batch is exactly half.
PRICE_IN_PER_M = 0.10
PRICE_OUT_PER_M = 0.60


def api_key() -> str:
    if os.environ.get("OPENAI_API_KEY"):
        return os.environ["OPENAI_API_KEY"]
    env = Path(".env")
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith("OPENAI_API_KEY="):
                return line.split("=", 1)[1].strip().strip("'\"")
    raise SystemExit("no OPENAI_API_KEY in the environment or .env")


def receipts() -> list[tuple[str, dict[str, Any]]]:
    out = []
    for path in sorted(RUNS.glob("batch_*.json")):
        out.append((path.name, json.loads(path.read_text(encoding="utf-8"))))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    headers = {"Authorization": f"Bearer {api_key()}"}
    rows: list[dict[str, Any]] = []
    with httpx.Client(timeout=120, headers=headers) as client:
        for name, info in receipts():
            batch_id = info.get("batch_id", "")
            row: dict[str, Any] = {
                "receipt": name,
                "arm": info.get("arm"),
                "stage": info.get("stage"),
                "requests": info.get("requests"),
                "batch_id": batch_id,
            }
            b = client.get(f"{BASE}/batches/{batch_id}")
            if b.status_code != 200:
                row["status"] = f"unreachable ({b.status_code})"
                rows.append(row)
                continue
            batch = b.json()
            row["status"] = batch.get("status")
            file_id = batch.get("output_file_id")
            if not file_id:
                row["note"] = "no output file"
                rows.append(row)
                continue
            content = client.get(f"{BASE}/files/{file_id}/content")
            if content.status_code != 200:
                row["note"] = f"output aged out ({content.status_code})"
                rows.append(row)
                continue
            tin = tout = 0
            for line in content.text.splitlines():
                usage = json.loads(line).get("response", {}).get("body", {}).get("usage", {})
                tin += int(usage.get("prompt_tokens", 0) or 0)
                tout += int(usage.get("completion_tokens", 0) or 0)
            row |= {
                "prompt_tokens": tin,
                "completion_tokens": tout,
                "usd": round(tin / 1e6 * PRICE_IN_PER_M + tout / 1e6 * PRICE_OUT_PER_M, 6),
            }
            rows.append(row)

    priced = [r for r in rows if "usd" in r]
    total = {
        "batches": len(rows),
        "priced": len(priced),
        "unavailable": len(rows) - len(priced),
        "requests": sum(int(r.get("requests") or 0) for r in priced),
        "prompt_tokens": sum(int(r["prompt_tokens"]) for r in priced),
        "completion_tokens": sum(int(r["completion_tokens"]) for r in priced),
        "usd_at_batch_rates": round(sum(float(r["usd"]) for r in priced), 4),
        "ceiling_usd": 5.00,
    }
    if args.json:
        print(json.dumps({"batches": rows, "total": total}, indent=2))
        return 0
    for r in rows:
        tail = (
            f"{r['prompt_tokens']:>8,} in {r['completion_tokens']:>6,} out  ${r['usd']:.5f}"
            if "usd" in r
            else f"  {r.get('note', r.get('status'))}"
        )
        print(f"{r['receipt']:46} {r['status']!s:11} {r.get('requests')!s:>4} req  {tail}")
    print(
        f"\n{total['priced']} of {total['batches']} batches priced "
        f"({total['unavailable']} unavailable); {total['requests']} requests; "
        f"{total['prompt_tokens']:,} in / {total['completion_tokens']:,} out; "
        f"**${total['usd_at_batch_rates']:.4f}** at batch rates against the $5.00 ceiling"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
