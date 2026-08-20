"""Measure whether pinning the thinking budget restores deterministic output.

`gemini-3.5-flash-lite` ignores `temperature` (observed: "uses fixed sampling defaults;
the sampling parameter(s) temperature will be ignored"), so v2.1's greedy-decoding
assumption no longer holds. Thinking is also the dominant cost term, because the output
rate bills thinking tokens.

This probe runs one fixed prompt N times at several thinking budgets and reports how many
distinct outputs came back, plus the token split. Phase 4's variance handling depends on
the answer, so it is measured rather than assumed.

Usage:
    python scripts/determinism_probe.py --runs 5
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_settings

PROMPT = (
    "In exactly three sentences, explain what low-rank adaptation (LoRA) changes about "
    "fine-tuning a transformer."
)
SYSTEM = "You are a precise technical writer. Answer in exactly three sentences."


async def probe(setting: str, runs: int) -> dict[str, object]:
    """`setting` is 'default', an integer thinking budget, or 'effort:<level>'."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    settings = get_settings()
    kwargs: dict[str, object] = {
        "model": settings.model_name(),
        "api_key": settings.google_api_key.get_secret_value(),
        "temperature": 0.0,
    }
    if setting.startswith("effort:"):
        kwargs["reasoning_effort"] = setting.split(":", 1)[1]
    elif setting != "default":
        kwargs["thinking_budget"] = int(setting)

    llm = ChatGoogleGenerativeAI(**kwargs)  # type: ignore[arg-type]

    digests: list[str] = []
    thinking: list[int] = []
    output: list[int] = []
    sample = ""
    for _ in range(runs):
        resp = await llm.ainvoke([("system", SYSTEM), ("user", PROMPT)])
        text = str(resp.text).strip()
        sample = sample or text
        digests.append(hashlib.sha256(text.encode()).hexdigest()[:10])
        meta = getattr(resp, "usage_metadata", None) or {}
        details = meta.get("output_token_details") or {}
        thinking.append(int(details.get("reasoning", 0)))
        output.append(int(meta.get("output_tokens", 0)))

    return {
        "setting": setting,
        "runs": runs,
        "distinct": len(set(digests)),
        "thinking_tokens": thinking,
        "output_tokens": output,
        "sample": sample[:110],
    }


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument(
        "--budgets",
        type=str,
        default="default,effort:minimal,effort:low,128",
        help="Comma-separated: 'default', an integer thinking budget, or 'effort:<level>'.",
    )
    args = parser.parse_args()

    budgets: list[str] = [b.strip() for b in args.budgets.split(",")]

    print(f"model : {get_settings().model_name()}")
    print(f"prompt: {PROMPT}\n")
    print(f"{'setting':<16} {'distinct':<10} {'mean think':<12} {'mean out':<10} verdict")
    print("-" * 78)

    for setting in budgets:
        try:
            r = await probe(setting, args.runs)
        except Exception as exc:  # an unsupported setting is a result, not a crash
            print(f"{setting:<16} FAILED: {str(exc)[:56]}")
            continue
        think = r["thinking_tokens"]
        out = r["output_tokens"]
        verdict = "deterministic" if r["distinct"] == 1 else f"{r['distinct']} distinct outputs"
        print(
            f"{setting:<16} {f'{r["distinct"]}/{r["runs"]}':<10} "
            f"{sum(think) / len(think):<12.0f} {sum(out) / len(out):<10.0f} {verdict}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
