"""The landing page served at `GET /`: one static HTML file, no framework, no external requests.

The page is loaded once. Its example question is substituted from `EXAMPLE_QUESTION` — the one
copy; the README's curl, which `make smoke-live` runs, must carry the same text (tested) — and a
Content-Security-Policy is built from the SHA-256 of the page's inline script and style, with
`connect-src 'self'`. The browser, not the page's good behaviour, is what keeps it from
contacting any other origin. `frame-ancestors` admits huggingface.co, which shows a Space's app
in an iframe on its own page.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path

# The README's smoke-live question: judged a correct, cited answer in all seven judged runs of
# the frozen set (item sp-001), and verified on the live Space (DECISIONS D-061).
EXAMPLE_QUESTION = (
    "What is the top-1 error rate achieved by LPA (mean + varied bound) using ResNet-110 on the "
    "CIFAR-100 dataset?"
)

PAGE = Path(__file__).resolve().parent / "static" / "index.html"
_INLINE = re.compile(r"<(script|style)>(.*?)</\1>", re.DOTALL)


def _hash(source: str) -> str:
    digest = hashlib.sha256(source.encode("utf-8")).digest()
    return "'sha256-" + base64.b64encode(digest).decode("ascii") + "'"


@lru_cache(maxsize=1)
def page() -> tuple[str, str]:
    """(html, content-security-policy). Raises if the file is missing — loud, not a blank page."""
    # Substituted as a JSON string literal, `<` escaped, so no question text can close the script.
    literal = json.dumps(EXAMPLE_QUESTION).replace("<", "\\u003c")
    html = PAGE.read_text(encoding="utf-8").replace('"__EXAMPLE_QUESTION__"', literal)
    if "__EXAMPLE_QUESTION__" in html:
        raise RuntimeError(f"{PAGE}: the example placeholder was not substituted")
    hashes: dict[str, list[str]] = {"script": [], "style": []}
    for kind, body in _INLINE.findall(html):
        hashes[kind].append(_hash(body))
    csp = "; ".join(
        [
            "default-src 'none'",
            "script-src " + " ".join(hashes["script"]),
            "style-src " + " ".join(hashes["style"]),
            "connect-src 'self'",
            "img-src 'none'",
            "base-uri 'none'",
            "form-action 'none'",
            "frame-ancestors 'self' https://huggingface.co",
        ]
    )
    return html, csp
