"""Which license does each corpus paper carry on arXiv? An audit — it changes nothing.

    make corpus-licenses            # queries arXiv's OAI-PMH (public, ~3 s/request, ~8 min)
    make corpus-licenses REPORT=1   # reprint from data/LICENSES.json

The corpus is 150 arXiv papers whose full text is chunked into ``data/chunks_512.json`` and
served back, in excerpts, by a public endpoint. What that is permitted to do depends on each
paper's license, which ``data/metadata.json`` never recorded. arXiv's OAI-PMH ``arXiv`` format
carries it (``<license>``); a paper with no ``<license>`` element was submitted under arXiv's
default non-exclusive distribution license, which grants arXiv — not third parties — the right
to distribute it. This script records the per-paper license and the counts; interpreting them
is a decision for the README, not for code.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
import time
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "data" / "LICENSES.json"
OAI = "https://oaipmh.arxiv.org/oai"
DEFAULT = "none recorded (arXiv default non-exclusive distribution license)"
LICENSE_RE = re.compile(r"<license>\s*(.*?)\s*</license>", re.S)

NAMES = {
    "http://creativecommons.org/licenses/by/4.0/": "CC BY 4.0",
    "http://creativecommons.org/licenses/by-sa/4.0/": "CC BY-SA 4.0",
    "http://creativecommons.org/licenses/by-nc-sa/4.0/": "CC BY-NC-SA 4.0",
    "http://creativecommons.org/licenses/by-nc-nd/4.0/": "CC BY-NC-ND 4.0",
    "http://creativecommons.org/licenses/by-nd/4.0/": "CC BY-ND 4.0",
    "http://creativecommons.org/licenses/by-nc/4.0/": "CC BY-NC 4.0",
    "http://creativecommons.org/publicdomain/zero/1.0/": "CC0 1.0",
    "http://arxiv.org/licenses/nonexclusive-distrib/1.0/": "arXiv non-exclusive distribution",
}


def fetch(paper_id: str, client: httpx.Client) -> str:
    for attempt in range(4):
        r = client.get(
            OAI,
            params={
                "verb": "GetRecord",
                "identifier": f"oai:arXiv.org:{paper_id}",
                "metadataPrefix": "arXiv",
            },
        )
        if r.status_code == 503:  # arXiv's documented back-off signal
            time.sleep(int(r.headers.get("Retry-After", "10")) + attempt * 5)
            continue
        r.raise_for_status()
        if "<error" in r.text and "idDoesNotExist" in r.text:
            return "not found on arXiv"
        match = LICENSE_RE.search(r.text)
        return match.group(1).strip() if match else DEFAULT
    raise RuntimeError(f"{paper_id}: arXiv kept answering 503")


def report(data: Mapping[str, object]) -> None:
    papers = data["papers"]
    assert isinstance(papers, dict)
    counts = Counter(NAMES.get(str(v), str(v)) for v in papers.values())
    print(f"Corpus licenses on arXiv — {len(papers)} papers, fetched {data['fetched_utc']}\n")
    for name, n in counts.most_common():
        print(f"  {n:>4}  {name}")


ATTRIBUTION = REPO / "CORPUS_ATTRIBUTION.md"


def render_attribution() -> str:
    """CORPUS_ATTRIBUTION.md: every corpus paper with its arXiv id, title, authors and license.

    Rendered from data/LICENSES.json and data/metadata.json — never edited by hand;
    tests/test_docs.py fails a stale copy.
    """
    data = json.loads(OUT.read_text(encoding="utf-8"))
    meta = {
        str(p["paper_id"]): p
        for p in json.loads((REPO / "data" / "metadata.json").read_text(encoding="utf-8"))
    }

    def cell(text: str) -> str:
        return " ".join(str(text).split()).replace("|", "\\|")

    rows = []
    for pid in sorted(data["papers"]):
        paper = meta[pid]
        authors = paper.get("authors") or []
        names = ", ".join(str(a) for a in authors)  # all of them: CC BY credits the creators
        lic = NAMES.get(str(data["papers"][pid]), str(data["papers"][pid]))
        url = str(data["papers"][pid])
        lic_cell = f"[{lic}]({url})" if url.startswith("http") else lic
        rows.append(
            f"| [{pid}](https://arxiv.org/abs/{pid}) | {cell(paper.get('title', ''))} | "
            f"{cell(names)} | {lic_cell} |"
        )
    return (
        "# Corpus attribution\n\n"
        "<!-- Rendered by `make corpus-licenses ATTRIBUTION=1` from data/LICENSES.json and "
        "data/metadata.json. Do not edit by hand. -->\n\n"
        f"This project's corpus is the text of the {len(rows)} arXiv papers below, chunked into "
        "`data/chunks_512.json` and quoted in excerpts by the service. Each paper remains under "
        "the license its authors chose on arXiv (fetched "
        f"{data['fetched_utc'][:10]} from arXiv's OAI-PMH interface); this project's MIT license "
        "covers its code only. To request removal of a paper, see the README's License "
        "section.\n\n"
        "| arXiv id | Title | Authors | License |\n|---|---|---|---|\n" + "\n".join(rows) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--attribution", action="store_true", help="write CORPUS_ATTRIBUTION.md")
    args = parser.parse_args()
    if args.attribution:
        ATTRIBUTION.write_text(render_attribution(), encoding="utf-8")
        print(f"wrote {ATTRIBUTION}")
        return
    if args.report:
        report(json.loads(OUT.read_text(encoding="utf-8")))
        return
    ids = [str(p["paper_id"]) for p in json.loads((REPO / "data" / "metadata.json").read_text())]
    papers: dict[str, str] = {}
    with httpx.Client(timeout=30, follow_redirects=True) as client:
        for n, pid in enumerate(ids, 1):
            papers[pid] = fetch(pid, client)
            print(
                f"  {n:>3}/{len(ids)} {pid} {NAMES.get(papers[pid], papers[pid])}", file=sys.stderr
            )
            time.sleep(3)  # arXiv's API etiquette: one request every three seconds
    data = {
        "source": OAI + " (metadataPrefix=arXiv, <license>)",
        "fetched_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "papers": papers,
    }
    OUT.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    report(data)


if __name__ == "__main__":
    main()
