"""The settable embedding model, index name and query prefix default to the frozen constants.

D-002 freezes the retrieval components. Making them settable for the Phase 4 embedding
comparison arm must not move the shipping values, so the defaults are asserted equal to the
constants ported verbatim from v2.1 — and an env override reaches the objects that read it.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import BGE_QUERY_PREFIX, EMBEDDING_MODEL, INDEX_NAME, RetrievalSettings


class TestFrozenDefaults:
    def test_defaults_are_the_frozen_constants(self) -> None:
        r = RetrievalSettings()

        assert r.embedding_model == EMBEDDING_MODEL == "BAAI/bge-large-en"
        assert r.index_name == INDEX_NAME == "BGE_cs512"
        assert r.query_prefix == BGE_QUERY_PREFIX

    def test_the_comparison_arm_can_point_at_a_second_index(self, monkeypatch) -> None:
        from src.config import Settings

        monkeypatch.setenv("RETRIEVAL__EMBEDDING_MODEL", "BAAI/bge-small-en")
        monkeypatch.setenv("RETRIEVAL__INDEX_NAME", "BGEsmall_cs512")

        s = Settings()

        assert s.retrieval.embedding_model == "BAAI/bge-small-en"
        assert s.retrieval.index_name == "BGEsmall_cs512"
