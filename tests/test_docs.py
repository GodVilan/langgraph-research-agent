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

REPO = Path(__file__).resolve().parent.parent
README = REPO / "README.md"

STATS_BLOCK = re.compile(r"<!-- STATS:START -->(.*?)<!-- STATS:END -->", re.DOTALL)


def readme() -> str:
    return README.read_text(encoding="utf-8")


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

    DOCS: ClassVar[list[Path]] = [
        Path(__file__).resolve().parent.parent / p
        for p in ("README.md", "CLAUDE.md", "docs/EVALS.md", "docs/DECISIONS.md")
    ]

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
