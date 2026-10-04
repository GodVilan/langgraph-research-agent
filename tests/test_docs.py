"""Guards on the documentation's own claims.

The repo's credibility argument is that every number in it is regenerable by a command.
Three different test counts in one README would discredit the figures that *are* correct,
so the one number most likely to drift gets a test.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import ClassVar

import pytest

REPO = Path(__file__).resolve().parent.parent
README = REPO / "README.md"

STATS_BLOCK = re.compile(r"<!-- STATS:START -->(.*?)<!-- STATS:END -->", re.DOTALL)


def readme() -> str:
    return README.read_text(encoding="utf-8")


def tracked(rel: str) -> bool:
    """Whether ``rel`` is part of the repository, as CI sees it.

    In a git checkout, what git tracks. In an export (`make ci-local` runs the suite in a
    `git archive` of HEAD, with no .git), every file present *is* tracked by construction. A
    docs test that reads an untracked file passes on the machine holding it and fails on every
    runner: this test's CLAUDE.md read did exactly that from run #9 on (D-056).
    """
    import subprocess

    if not (REPO / ".git").exists():
        return (REPO / rel).is_file()
    return (
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", rel], cwd=REPO, capture_output=True
        ).returncode
        == 0
    )


class TestReadmeStats:
    def test_generated_block_is_present(self) -> None:
        assert STATS_BLOCK.search(readme()), "run `make readme-stats`"

    def test_block_reports_a_test_count(self) -> None:
        block = STATS_BLOCK.search(readme())
        assert block is not None
        assert re.search(r"\|\s*Tests\s*\|\s*\d+", block.group(1)), (
            "the stats block should carry a measured test count"
        )

    def test_no_hardcoded_test_count_outside_the_generated_block(self) -> None:
        """The C-1 failure: a count typed into prose drifts away from reality silently."""
        prose = STATS_BLOCK.sub("", readme())
        offenders = re.findall(r"\b\d{2,4}\s+tests\b", prose)
        assert not offenders, (
            f"hardcoded test count(s) in README prose: {offenders}. "
            f"Cite `make readme-stats` output instead."
        )

    def test_stats_block_matches_the_collected_test_count(self) -> None:
        """Fails when the block is stale. Regenerate with `make readme-stats`."""
        import subprocess

        block = STATS_BLOCK.search(readme())
        assert block is not None
        claimed = int(re.search(r"\|\s*Tests\s*\|\s*(\d+)", block.group(1)).group(1))  # type: ignore[union-attr]

        # `sys.executable`, not `.venv/bin/python`: the hardcoded path exists only on a
        # machine laid out like the developer's, so in CI this raised FileNotFoundError and
        # the guard failed rather than checking anything. Found by the first CI run
        # (DECISIONS D-022 — a working local environment is not evidence of a portable one).
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "--collect-only", "-q"],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=False,
        )
        match = re.search(r"(\d+) tests collected", proc.stdout)
        # Deliberately not `pytest.skip`. A guard that skips when it cannot run is a guard
        # that disappears exactly when something is wrong — which is what happened in CI,
        # where the FileNotFoundError above meant this check had never once executed there.
        assert match is not None, (
            f"could not collect tests in a subprocess (exit {proc.returncode}).\n"
            f"stdout: {proc.stdout[-500:]}\nstderr: {proc.stderr[-500:]}"
        )
        assert claimed == int(match.group(1)), (
            f"README claims {claimed} tests, pytest collects {match.group(1)}. "
            f"Run `make readme-stats`."
        )


class TestPairedMetricsInTheReadme:
    """Two figures that mislead alone are only allowed together, enforced here.

    Recall@k measured on v3 is v2.1's frozen retriever (D-002) against paraphrased questions;
    printed alone a reader takes it as v3's retrieval quality. It is interpretable only beside
    the v2.1 re-run on the same set. Same shape as the refusal pair: refusal accuracy alone is
    the near-perfect-metric trap.
    """

    README = Path(__file__).resolve().parent.parent / "README.md"

    def _lines_with(self, pattern: str) -> list[str]:
        import re

        text = self.README.read_text(encoding="utf-8")
        return [line for line in text.splitlines() if re.search(pattern, line)]

    def test_no_recall_figure_without_the_v21_figure_beside_it(self) -> None:
        import re

        offending = [
            line
            for line in self._lines_with(r"(?i)\b(recall@\d+|mrr)\b\s*[:=]?\s*0?\.\d")
            if not re.search(r"(?i)v2\.1", line)
        ]
        assert not offending, (
            "a Recall@k / MRR figure appears in the README without the v2.1 figure on the "
            f"same line: {offending}"
        )

    def test_no_refusal_accuracy_without_hallucinated_refusal_beside_it(self) -> None:
        import re

        offending = [
            line
            for line in self._lines_with(r"(?i)refusal accuracy\s*\**\s*\d+\s*(/|of)\s*\d+")
            if not re.search(r"(?i)hallucinated[- ]refusal", line)
        ]
        assert not offending, f"refusal accuracy printed without its pair: {offending}"


class TestPerfectAgreementCarriesItsAmendment:
    """A post-amendment perfect agreement is never quoted without the amendment beside it.

    `make judge-agreement` emits the clause on the line; this covers the four other paths to
    the number — README, EVALS.md, DECISIONS.md, CLAUDE.md — because the untagged copy is the
    one that gets quoted. "25 of 25" between arms was true before the amendment and is
    allowed with "each other" on the line; "25 of 25" with the human was not, and needs
    "amendment".
    """

    # CLAUDE.md is the local operating brief and is not tracked, so it is not checked here: a
    # test of a file the repository does not contain cannot pass on a clean checkout (D-056).
    CANDIDATES: ClassVar[tuple[str, ...]] = (
        "README.md",
        "CLAUDE.md",
        "docs/EVALS.md",
        "docs/DECISIONS.md",
        "docs/POSTMORTEM.md",
    )
    DOCS: ClassVar[list[Path]] = [REPO / p for p in CANDIDATES if tracked(p)]

    def test_the_tracked_docs_are_the_ones_checked(self) -> None:
        """Filtering to tracked files must not quietly empty the list."""
        names = {p.relative_to(REPO).as_posix() for p in self.DOCS}
        assert {"README.md", "docs/EVALS.md", "docs/DECISIONS.md"} <= names
        assert "CLAUDE.md" not in names
        assert all(p.is_file() for p in self.DOCS)

    def test_every_25_of_25_line_names_the_amendment_or_is_cross_arm(self) -> None:
        import re

        offending: list[str] = []
        for doc in self.DOCS:
            for n, line in enumerate(doc.read_text(encoding="utf-8").splitlines(), start=1):
                if not re.search(r"\b25 of 25\b", line):
                    continue
                if re.search(r"(?i)amendment|each other", line):
                    continue
                offending.append(f"{doc.name}:{n}: {line.strip()[:100]}")
        assert not offending, (
            "perfect agreement quoted without the rubric-v2 clause:\n" + "\n".join(offending)
        )


class TestGeneratedStatusAndSpend:
    """C-1 recurred: "Status: Phase 3 of 5" and "$0.00 spent" survived Phase 4's close as prose.

    Both now have one source each — docs/status.json and evals/runs/judge_spend.json — and
    are rendered by `make readme-stats`. These fail a stale rendering, and a spend figure
    typed anywhere outside the rendered blocks.
    """

    DOCS: ClassVar[list[Path]] = [REPO / "README.md", REPO / "docs" / "BUDGET.md"]

    @staticmethod
    def _block(text: str, name: str) -> str:
        match = re.search(rf"<!-- {name}:START -->.*?<!-- {name}:END -->", text, re.DOTALL)
        assert match is not None, f"missing {name} markers"
        return match.group(0)

    def test_the_ci_badge_sits_beside_the_live_url_and_points_at_this_workflow(self) -> None:
        import json

        status = json.loads((REPO / "docs" / "status.json").read_text(encoding="utf-8"))
        block = self._block(readme(), "STATUS")
        line = next(ln for ln in block.splitlines() if "**Live:" in ln)
        workflow = f"https://github.com/{status['repo']}/actions/workflows/ci.yml"
        assert f"({workflow}/badge.svg?branch=main)" in line

    def test_the_readme_status_is_the_rendered_one(self) -> None:
        sys.path.insert(0, str(REPO))
        from scripts.readme_stats import render_status

        assert self._block(readme(), "STATUS") == render_status(), "run `make readme-stats`"

    def test_every_spend_block_is_the_rendered_one(self) -> None:
        sys.path.insert(0, str(REPO))
        from scripts.readme_stats import render_spend

        for doc in self.DOCS:
            text = doc.read_text(encoding="utf-8")
            assert self._block(text, "JUDGESPEND") == render_spend(), f"{doc.name}: stale spend"

    def test_the_per_line_spend_table_is_the_rendered_one(self) -> None:
        sys.path.insert(0, str(REPO))
        from scripts.readme_stats import render_spend_table

        budget = (REPO / "docs" / "BUDGET.md").read_text(encoding="utf-8")
        assert self._block(budget, "SPENDTABLE") == render_spend_table(), "run `make readme-stats`"

    def test_no_openai_spend_figure_is_typed_outside_the_blocks(self) -> None:
        offenders: list[str] = []
        for doc in self.DOCS:
            text = re.sub(
                r"<!-- (?:JUDGE)?SPEND:START -->.*?<!-- (?:JUDGE)?SPEND:END -->|"
                r"<!-- GEMINI:START -->.*?<!-- GEMINI:END -->",
                "",
                doc.read_text(encoding="utf-8"),
                flags=re.DOTALL,
            )
            for line in text.splitlines():
                if (
                    re.search(r"(?i)(spent|spend)\b", line)
                    and re.search(r"(?i)openai|judge", line)
                    and re.search(r"\$0\.\d{3,4}\b", line)
                ):
                    offenders.append(f"{doc.name}: {line.strip()[:100]}")
        assert not offenders, "typed spend figure(s):\n" + "\n".join(offenders)

    def test_a_live_url_and_the_readme_curl_agree(self) -> None:
        import json

        status = json.loads((REPO / "docs" / "status.json").read_text(encoding="utf-8"))
        curl = re.search(r"<!-- CURL:START -->.*?<!-- CURL:END -->", readme(), re.DOTALL)
        assert curl is not None
        if status["live_url"]:
            assert status["live_url"].rstrip("/") in curl.group(0), (
                "the README's curl must target the live URL once there is one"
            )


class TestGeneratedBlocksAreUnique:
    """Two generators once shared the SPEND marker, and `readme-stats` overwrote `make
    budget`'s trace table in BUDGET.md with the OpenAI line; the staleness test checked only
    the first block of each name, so it passed. Each generated block now appears at most
    once per file, under a marker one generator owns."""

    def test_no_marker_appears_twice_in_a_file(self) -> None:
        for doc in (REPO / "README.md", REPO / "docs" / "BUDGET.md"):
            names = re.findall(r"<!-- ([A-Z]+):START -->", doc.read_text(encoding="utf-8"))
            dupes = sorted({n for n in names if names.count(n) > 1})
            assert not dupes, f"{doc.name}: marker(s) used twice: {dupes}"

    def test_the_trace_table_and_the_judge_line_are_different_blocks(self) -> None:
        budget = (REPO / "docs" / "BUDGET.md").read_text(encoding="utf-8")
        trace = re.search(r"<!-- SPEND:START -->.*?<!-- SPEND:END -->", budget, re.S)
        judge = re.search(r"<!-- JUDGESPEND:START -->.*?<!-- JUDGESPEND:END -->", budget, re.S)
        assert trace is not None and judge is not None
        assert "Generated by `make budget`" in trace.group(0)
        assert "OpenAI judge spend" in judge.group(0)


class TestGeminiBilledIsNeverAssumed:
    """D-046: every "Gemini billed $0" came from a free-tier assumption on a key that was
    billed $7.60. A billed figure is the provider's record or "unverified", never 0 by default."""

    DOCS: ClassVar[list[Path]] = [
        REPO / p
        for p in (
            "README.md",
            "docs/BUDGET.md",
            "docs/OBSERVABILITY.md",
            "docs/SERVING.md",
            "docs/PHASE4.md",
        )
    ]

    def test_the_gemini_block_is_the_rendered_one(self) -> None:
        sys.path.insert(0, str(REPO))
        from scripts.readme_stats import render_gemini

        for doc in (REPO / "README.md", REPO / "docs" / "BUDGET.md"):
            block = re.search(
                r"<!-- GEMINI:START -->.*?<!-- GEMINI:END -->",
                doc.read_text(encoding="utf-8"),
                re.S,
            )
            assert block is not None and block.group(0) == render_gemini(), doc.name

    def test_no_doc_asserts_gemini_was_billed_zero(self) -> None:
        # A line may *quote* the old claim to say it was wrong; it may not make it.
        claim = re.compile(r"(?i)(\$0(\.0+)?\s*billed|billed[^.]{0,40}\$0(\.0+)?\b(?![.\d]))")
        retraction = re.compile(r"(?i)used to|was not|assum|wrong|D-046|no longer|never as")
        offending = []
        for doc in self.DOCS:
            for n, line in enumerate(doc.read_text(encoding="utf-8").splitlines(), 1):
                if claim.search(line) and not retraction.search(line):
                    offending.append(f"{doc.name}:{n}: {line.strip()[:110]}")
        assert not offending, "Gemini billed-$0 claim(s):\n" + "\n".join(offending)


class TestCorpusLicensesAreRendered:
    def test_the_license_counts_are_the_rendered_ones(self) -> None:
        sys.path.insert(0, str(REPO))
        from scripts.readme_stats import render_licenses

        block = re.search(r"<!-- LICENSES:START -->.*?<!-- LICENSES:END -->", readme(), re.S)
        assert block is not None and block.group(0) == render_licenses()

    def test_every_corpus_paper_has_a_recorded_license(self) -> None:
        import json

        licenses = json.loads((REPO / "data" / "LICENSES.json").read_text(encoding="utf-8"))
        papers = json.loads((REPO / "data" / "metadata.json").read_text(encoding="utf-8"))
        assert {str(p["paper_id"]) for p in papers} == set(licenses["papers"])


class TestLoadCheckIsRendered:
    """The deployed load-check figures come from the artifact, never typed (D-052)."""

    def test_the_load_check_and_uncited_blocks_are_the_rendered_ones(self) -> None:
        sys.path.insert(0, str(REPO))
        from scripts.readme_stats import render_loadcheck, render_uncited

        for name, render in (("LOADCHECK", render_loadcheck), ("UNCITED", render_uncited)):
            block = re.search(rf"<!-- {name}:START -->.*?<!-- {name}:END -->", readme(), re.S)
            assert block is not None and block.group(0) == render(), name

    def test_an_invalid_load_check_cannot_be_rendered(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import json

        sys.path.insert(0, str(REPO))
        import scripts.readme_stats as rs

        slept = REPO / "evals" / "runs" / "loadcheck_deployed-client-slept.json"
        monkeypatch.setattr(rs, "LOADCHECK_JSON", slept)
        with pytest.raises(SystemExit, match="invalid"):
            rs.render_loadcheck()
        assert json.loads(slept.read_text(encoding="utf-8"))["invalid"]

    def test_a_single_user_run_that_slept_or_dropped_a_request_cannot_be_rendered(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import json

        sys.path.insert(0, str(REPO))
        import scripts.readme_stats as rs

        first, second = rs.SINGLE_USER_RUNS
        base = json.loads(first.read_text(encoding="utf-8"))
        for doctor, expect in (
            (lambda d: d["records"][0].__setitem__("clock_drift_s", 900.0), "invalid"),
            (lambda d: d["records"][0].__setitem__("status", 503), "not served"),
            (lambda d: d.pop("space_commit"), "does not record the Space"),
        ):
            data = json.loads(json.dumps(base))
            doctor(data)
            copy = tmp_path / "single.json"
            copy.write_text(json.dumps(data), encoding="utf-8")
            # Either run, doctored, blocks the render: the good one does not cover for it.
            monkeypatch.setattr(rs, "SINGLE_USER_RUNS", (second, copy))
            with pytest.raises(SystemExit, match=expect):
                rs.render_loadcheck()

    def test_the_single_user_rows_are_the_locked_runs_and_the_old_row_is_gone(self) -> None:
        readme = (REPO / "README.md").read_text(encoding="utf-8")
        block = readme.split("<!-- LOADCHECK:START -->", 1)[1].split("<!-- LOADCHECK:END -->")[0]
        rows = [ln for ln in block.splitlines() if ln.startswith("| Single user")]
        assert len(rows) == 2 and all("Space `d96cd36e`" in r for r in rows)
        assert all("(slowest of 10)" in r for r in rows)
        assert "7745886e" not in block  # history, in DECISIONS D-064
        decisions = (REPO / "docs" / "DECISIONS.md").read_text(encoding="utf-8")
        assert "p50 7.1 s, p95 14.3 s" in decisions and "7745886e" in decisions

    def test_an_unread_uncited_response_blocks_the_render(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import json

        sys.path.insert(0, str(REPO))
        import scripts.readme_stats as rs

        data = json.loads(rs.LOADCHECK_JSON.read_text(encoding="utf-8"))
        data["uncited_reading"]["records"].popitem()
        copy = tmp_path / "lc.json"
        copy.write_text(json.dumps(data), encoding="utf-8")
        monkeypatch.setattr(rs, "LOADCHECK_JSON", copy)
        with pytest.raises(SystemExit, match="not yet read"):
            rs.render_uncited()


class TestCloudDashboardCostIsNotional:
    """D-060: Langfuse Cloud's cost is computed from Langfuse's own list prices — a notional
    figure, never a charge. A published line quoting it must say so."""

    DOCS: ClassVar[list[Path]] = [
        REPO / "README.md",
        *sorted((REPO / "docs").glob("*.md")),
    ]

    def test_a_langfuse_cloud_dollar_figure_is_labelled_notional(self) -> None:
        cloud = re.compile(r"(?i)langfuse cloud|cloud\.langfuse\.com")
        dollars = re.compile(r"\$\d")
        offending = [
            f"{doc.name}:{n}: {line.strip()[:110]}"
            for doc in self.DOCS
            if doc.name != "DECISIONS.md"  # the record of how figures were found
            for n, line in enumerate(doc.read_text(encoding="utf-8").splitlines(), 1)
            if cloud.search(line) and dollars.search(line) and "notional" not in line.lower()
        ]
        assert not offending, "Langfuse Cloud cost quoted without 'notional':\n" + "\n".join(
            offending
        )


class TestCiClaimIsExact:
    """D-050: CI replays committed run artifacts through the gate; it never runs the agent."""

    def test_the_readme_says_what_ci_does(self) -> None:
        assert "does not run the agent on the pushed code" in readme()

    def test_no_doc_calls_the_gate_live_or_says_ci_evaluates_the_code(self) -> None:
        pattern = re.compile(r"(?i)live regression gate|CI (evaluates|runs the eval)")
        offenders = []
        for doc in (REPO / "README.md", REPO / "docs" / "EVALS.md", REPO / "docs" / "PHASE4.md"):
            for n, line in enumerate(doc.read_text(encoding="utf-8").splitlines(), 1):
                if pattern.search(line):
                    offenders.append(f"{doc.name}:{n}: {line.strip()[:100]}")
        assert not offenders, "\n".join(offenders)


class TestCorpusAttribution:
    def test_the_attribution_file_is_the_rendered_one(self) -> None:
        sys.path.insert(0, str(REPO))
        from scripts.corpus_licenses import render_attribution

        text = (REPO / "CORPUS_ATTRIBUTION.md").read_text(encoding="utf-8")
        assert text == render_attribution(), "run `make corpus-licenses ATTRIBUTION=1`"

    def test_every_paper_is_credited(self) -> None:
        import json

        text = (REPO / "CORPUS_ATTRIBUTION.md").read_text(encoding="utf-8")
        for paper in json.loads((REPO / "data" / "metadata.json").read_text(encoding="utf-8")):
            assert f"[{paper['paper_id']}](https://arxiv.org/abs/{paper['paper_id']})" in text

    def test_the_readme_carries_a_takedown_route(self) -> None:
        assert "github.com/GodVilan/langgraph-research-agent/issues" in readme()


# ── The header: badges carry no metric and no status that is not live ────────────────────


def header_block() -> str:
    match = re.search(r"<!-- HEADER:START -->.*?<!-- HEADER:END -->", readme(), re.DOTALL)
    assert match is not None, "missing HEADER markers"
    return match.group(0)


def shields_parts(url: str) -> tuple[str, str]:
    """(label, message) of a static shields.io badge URL, unescaped; message may be empty."""
    path = url.removeprefix("https://img.shields.io/badge/")
    parts = [p.replace("\0", "-") for p in path.replace("--", "\0").split("-")]
    unescape = lambda t: t.replace("__", "\1").replace("_", " ").replace("\1", "_")  # noqa: E731
    if len(parts) == 2:
        return unescape(parts[0]), ""
    assert len(parts) == 3, f"not a static label-message-colour badge: {url}"
    return unescape(parts[0]), unescape(parts[1])


class TestTheHeader:
    """A badge is a claim in a picture. The header may carry a link to the Space, static stack
    badges, the licence and the base-image pin — nothing that reads as a measurement, and no
    status unless something renders it live (the CI badge, in the status line below)."""

    HF_BADGE = (
        "https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg"
    )
    # label -> what its message may be; None means no message at all.
    STATIC: ClassVar[dict[str, str | None]] = {
        "LangGraph": None,
        "FastAPI": None,
        "Langfuse": None,
        "python": "lock",
        "docker": "pinned by digest",
        "code": "MIT",
    }

    def test_the_header_is_the_rendered_one(self) -> None:
        sys.path.insert(0, str(REPO))
        from scripts.readme_stats import render_header

        assert header_block() == render_header(), "run `make readme-stats`"

    def test_the_title_and_the_header_open_the_readme(self) -> None:
        assert readme().startswith("# Scholium\n\n<!-- HEADER:START -->")
        assert readme().index("<!-- HEADER:END -->") < readme().index("<!-- STATUS:START -->")

    def test_the_paper_count_is_the_corpus_s_not_a_typed_one(self) -> None:
        import json

        papers = len(json.loads((REPO / "data" / "metadata.json").read_text(encoding="utf-8")))
        assert f"over {papers} arXiv ML papers" in header_block()

    def test_every_badge_is_allowed_and_carries_no_metric(self) -> None:
        import json

        sys.path.insert(0, str(REPO))
        from scripts.lock_check import lock_python

        python = lock_python(REPO / "requirements.lock")
        status = json.loads((REPO / "docs" / "status.json").read_text(encoding="utf-8"))
        images = re.findall(r"!\[[^\]]*\]\(([^)\s]+)\)", header_block())
        badges = [u for u in images if not u.startswith("docs/img/")]
        assert badges, "the header has no badge row"
        assert f"](https://huggingface.co/spaces/{status['space']})" in header_block()
        for url in badges:
            if url == self.HF_BADGE:
                continue
            assert url.startswith("https://img.shields.io/badge/") and "?" not in url, (
                f"{url}: only static shields badges — a dynamic one renders a status or a "
                f"number this test cannot vouch for"
            )
            label, message = shields_parts(url)
            assert label in self.STATIC, f"badge {label!r} is not on the allowed list"
            rule = self.STATIC[label]
            if rule is None:
                assert message == "", f"{label} badge carries a message: {message!r}"
            elif rule == "lock":
                assert message == python, f"python badge says {message}, the lock says {python}"
            else:
                assert message == rule, f"{label} badge says {message!r}"
            # The one number a badge may carry is the Python release.
            assert not re.search(r"\d|%", f"{label} {message}".replace(python, "")), url
        # The CI badge is live and belongs to the status line, not twice in the header.
        assert "badge.svg" not in header_block()

    def test_the_pinned_and_licence_badges_are_true(self) -> None:
        dockerfile = (REPO / "infra" / "Dockerfile").read_text(encoding="utf-8")
        assert re.search(r"^ARG PYTHON_IMAGE=\S+@sha256:[0-9a-f]{64}$", dockerfile, re.M)
        assert (REPO / "LICENSE").read_text(encoding="utf-8").startswith("MIT License")

    def test_every_worth_a_look_link_resolves(self) -> None:
        def slug(heading: str) -> str:
            text = re.sub(r"[^\w\- ]", "", heading.strip().lower())
            return text.replace(" ", "-")

        line = next(ln for ln in header_block().splitlines() if ln.startswith("**Worth a look"))
        links = re.findall(r"\[([^\]]+)\]\(([^)]+)\)", line)
        assert [name for name, _ in links] == [
            "Evaluation",
            "Where the rebuild is worse",
            "The spend finding",
            "Post-mortem",
        ]
        for _, target in links:
            path, _, anchor = target.partition("#")
            doc = REPO / path if path else README
            assert doc.exists(), target
            if anchor:
                headings = re.findall(r"^#+ (.+)$", doc.read_text(encoding="utf-8"), re.M)
                assert anchor in {slug(h) for h in headings}, f"{target}: no such heading"

    def test_the_screenshot_matches_its_record(self) -> None:
        import json
        import struct

        record = json.loads((REPO / "docs" / "img" / "landing.json").read_text(encoding="utf-8"))
        png = (REPO / "docs" / "img" / "landing.png").read_bytes()
        assert png[:8] == b"\x89PNG\r\n\x1a\n"
        width, height = struct.unpack(">II", png[16:24])
        css_w, css_h = record["viewport_css_px"]
        scale = record["device_scale_factor"]
        assert (width, height) == (css_w * scale, css_h * scale)
        assert record["cited_papers"] and set(record["cited_papers"]) <= set(
            record["sources_listed"]
        )
        assert record["captured_utc"][:16].replace("T", " ") in header_block()
