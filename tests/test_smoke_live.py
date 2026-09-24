"""The smoke test runs the README's example, so the example's markers must stay intact."""

from __future__ import annotations

from scripts.smoke_live import cited_source_papers, expected_index_sha, readme_curl


def test_the_readme_carries_one_runnable_curl_example() -> None:
    command = readme_curl()
    assert command.startswith("curl ")
    assert "/query" in command
    assert '"stream": false' in command  # JSON, so the smoke test can parse the answer


def test_citations_are_matched_to_returned_sources_in_every_form() -> None:
    sources = ["2605.30179", "2605.29580"]
    assert cited_source_papers("W0 [2605.30179_0006].", sources) == {"2605.30179"}
    assert cited_source_papers("[2605.30179_0006, 2605.29580_0003]", sources) == set(sources)
    # the form the live smoke test met: prefix dropped
    assert cited_source_papers("frozen [30179_0006, 29580_0003]", sources) == set(sources)
    assert not cited_source_papers("see paper 2605.30179", sources)
    assert not cited_source_papers("[2605.99999_0001]", sources)


def test_the_expected_index_checksum_is_the_committed_one() -> None:
    assert len(expected_index_sha()) == 64
