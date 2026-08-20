"""Guards on the documentation's own claims.

The repo's credibility argument is that every number in it is regenerable by a command.
Three different test counts in one README would discredit the figures that *are* correct,
so the one number most likely to drift gets a test.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

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

        proc = subprocess.run(
            [str(REPO / ".venv/bin/python"), "-m", "pytest", "--collect-only", "-q"],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=False,
        )
        match = re.search(r"(\d+) tests collected", proc.stdout)
        if match is None:
            pytest.skip("could not collect tests in a subprocess")
        assert claimed == int(match.group(1)), (
            f"README claims {claimed} tests, pytest collects {match.group(1)}. "
            f"Run `make readme-stats`."
        )
