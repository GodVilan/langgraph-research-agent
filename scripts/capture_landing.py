"""Screenshot the live landing page answering the README's example question.

    make landing-shot [URL=https://godvillain-scholium.hf.space]

Writes `docs/img/landing.png` and, beside it, `docs/img/landing.json` — where and when it was
taken, from which Space commit, with which browser and viewport, and which sources the answer
cited. `make readme-stats` renders the README caption from that record, so the caption is never
typed and cannot drift from the image.

Drives the system's Google Chrome, headless, with a fresh temporary profile (no one's sessions),
over the DevTools protocol through `websockets`, which the lock already pins — no browser
automation package is added for one picture. It asks the question the way a visitor does:
"Use the example question", then "Ask". The answer is unpinned (D-047), so a draw that cites no
returned source is not kept; it retries, at most three times, 20 s apart for the per-IP limiter.
One query per attempt reaches the live service and its Gemini key.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import datetime as dt
import itertools
import json
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import httpx
from websockets.asyncio.client import connect

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.smoke_live import cited_source_papers  # noqa: E402

PNG = REPO / "docs" / "img" / "landing.png"
RECORD = REPO / "docs" / "img" / "landing.json"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
SPACE = "godvillain/Scholium"
WIDTH, HEIGHT, SCALE = 1100, 900, 2
ATTEMPTS, SPACING_S, ANSWER_TIMEOUT_S = 3, 20.0, 150.0

# What the page shows once an answer has rendered: answer text, and the listed sources' ids.
READ_RESULT = """(() => {
  const answer = document.querySelector('#result .answer');
  const ids = [...document.querySelectorAll('#result ol li a')].map(a => a.textContent);
  const err = document.querySelector('#result .error');
  return JSON.stringify({answer: answer ? answer.textContent : null, sources: ids,
                         error: err ? err.textContent : null,
                         status: document.getElementById('status').textContent});
})()"""


class CDP:
    """The few DevTools calls this needs, over one page's websocket."""

    def __init__(self, ws: Any) -> None:
        self.ws, self.ids = ws, itertools.count(1)

    async def call(self, method: str, **params: Any) -> dict[str, Any]:
        msg_id = next(self.ids)
        await self.ws.send(json.dumps({"id": msg_id, "method": method, "params": params}))
        while True:
            reply = json.loads(await self.ws.recv())
            if reply.get("id") == msg_id:
                if "error" in reply:
                    raise RuntimeError(f"{method}: {reply['error']}")
                result: dict[str, Any] = reply.get("result", {})
                return result

    async def evaluate(self, expression: str) -> Any:
        out = await self.call("Runtime.evaluate", expression=expression, returnByValue=True)
        return out.get("result", {}).get("value")


def kept(result: dict[str, Any]) -> list[str]:
    """The cited papers that are also listed as returned sources; empty means discard the draw."""
    if not result.get("answer") or result.get("error"):
        return []
    return sorted(cited_source_papers(result["answer"], list(result.get("sources") or [])))


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def space_commit() -> str:
    r = httpx.get(f"https://huggingface.co/api/spaces/{SPACE}/runtime", timeout=30)
    r.raise_for_status()
    return str(r.json().get("sha") or "")[:8]


async def attempt(cdp: CDP, url: str) -> dict[str, Any]:
    await cdp.call("Page.navigate", url=f"{url}/")
    for _ in range(100):
        if await cdp.evaluate("document.readyState") == "complete":
            break
        await asyncio.sleep(0.1)
    await cdp.evaluate("document.getElementById('example').click()")
    await cdp.evaluate("document.getElementById('send').click()")
    deadline = time.monotonic() + ANSWER_TIMEOUT_S
    while time.monotonic() < deadline:
        result: dict[str, Any] = json.loads(await cdp.evaluate(READ_RESULT))
        if (result["answer"] or result["error"]) and not result["status"]:
            return result
        await asyncio.sleep(0.5)
    raise SystemExit(f"no answer rendered within {ANSWER_TIMEOUT_S:.0f} s")


def start_chrome(profile: str) -> tuple[subprocess.Popen[bytes], str, str]:
    """Headless Chrome on a free port with a fresh profile: (process, page websocket, version)."""
    port = free_port()
    chrome = subprocess.Popen(
        [
            CHROME,
            "--headless=new",
            f"--remote-debugging-port={port}",
            f"--user-data-dir={profile}",
            "--no-first-run",
            "--no-default-browser-check",
            "--hide-scrollbars",
            "about:blank",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    pages: list[dict[str, Any]] = []
    for _ in range(100):
        try:
            pages = httpx.get(f"http://127.0.0.1:{port}/json", timeout=2).json()
            if pages:
                break
        except httpx.HTTPError:
            pass
        time.sleep(0.1)
    browser = httpx.get(f"http://127.0.0.1:{port}/json/version", timeout=5).json()["Browser"]
    page_ws = next(p for p in pages if p.get("type") == "page")["webSocketDebuggerUrl"]
    return chrome, str(page_ws), str(browser)


async def capture(url: str, page_ws: str) -> dict[str, Any]:
    """Ask the example question until a draw cites a returned source, then screenshot it."""
    async with connect(page_ws, max_size=None) as ws:
        cdp = CDP(ws)
        await cdp.call(
            "Emulation.setDeviceMetricsOverride",
            width=WIDTH,
            height=HEIGHT,
            deviceScaleFactor=SCALE,
            mobile=False,
        )
        await cdp.call(
            "Emulation.setEmulatedMedia",
            features=[{"name": "prefers-color-scheme", "value": "light"}],
        )
        for n in range(1, ATTEMPTS + 1):
            result = await attempt(cdp, url)
            cited = kept(result)
            print(f"attempt {n}: {len(result.get('sources') or [])} source(s), cited {cited}")
            if cited:
                break
            if n == ATTEMPTS:
                raise SystemExit("no draw cited a returned source; nothing written")
            await asyncio.sleep(SPACING_S)
        when = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
        # To the bottom of the content (main's own bottom padding included), not the
        # viewport: a short page would otherwise carry a band of empty white below it.
        height = await cdp.evaluate(
            "Math.ceil(document.querySelector('main').getBoundingClientRect().bottom)"
        )
        shot = await cdp.call(
            "Page.captureScreenshot",
            format="png",
            captureBeyondViewport=True,
            clip={"x": 0, "y": 0, "width": WIDTH, "height": int(height), "scale": 1},
        )
    PNG.write_bytes(base64.b64decode(shot["data"]))
    return {
        "url": f"{url}/",
        "captured_utc": when,
        "viewport_css_px": [WIDTH, int(height)],
        "device_scale_factor": SCALE,
        "color_scheme": "light",
        "question": "the README's example question, via 'Use the example question'",
        "attempts": n,
        "sources_listed": result.get("sources"),
        "cited_papers": cited,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--url", default="https://godvillain-scholium.hf.space")
    args = parser.parse_args()
    profile = tempfile.mkdtemp(prefix="landing-shot-")
    chrome, page_ws, browser = start_chrome(profile)
    try:
        record = asyncio.run(capture(args.url.rstrip("/"), page_ws))
    finally:
        chrome.terminate()
        chrome.wait(timeout=10)
        shutil.rmtree(profile, ignore_errors=True)
    record = {
        "url": record.pop("url"),
        "captured_utc": record.pop("captured_utc"),
        "space_commit": space_commit(),
        "browser": f"{browser}, headless, fresh profile",
        **record,
    }
    RECORD.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {PNG.relative_to(REPO)} and {RECORD.relative_to(REPO)}: {record}")


if __name__ == "__main__":
    main()
