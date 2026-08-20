"""LangChain tools with Pydantic argument schemas.

These are real ``@tool`` objects — schema'd, bindable, and traceable — but *the model does
not choose between them*. The ``retrieve`` node selects a tool by rule (MIGRATION_MAP
§5.1). v2.1 let the model emit a tool name as a JSON string and then spent ~90 lines of
loop-guard machinery stopping it from choosing badly; making the choice structural deletes
that machinery rather than improving it.

Two things v2.1 got wrong that are fixed by construction here:

* ``compare_papers`` took ``"topic_a | topic_b | aspect"`` as one pipe-delimited string
  (v2.1 ``tools.py:104``) and returned a usage hint when parsing failed. It now has three
  typed fields.
* Tools returned a display string that was also the machine-readable result, so provenance
  had to be recovered by regex (AUDIT §4.19). These return structured results; formatting
  happens once, in ``finalize``.
"""

from __future__ import annotations

from src.agent.tools.retrieval_tools import (
    CompareArgs,
    FetchArxivArgs,
    KeywordSearchArgs,
    SearchCorpusArgs,
    build_retrieval_tools,
)

__all__ = [
    "CompareArgs",
    "FetchArxivArgs",
    "KeywordSearchArgs",
    "SearchCorpusArgs",
    "build_retrieval_tools",
]
