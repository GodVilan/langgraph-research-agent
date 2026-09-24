"""Every check has a call site, or it fails here rather than in a report.

D-026: two checks written in response to rejections were reported as fixes and never ran.
`papers_contributing_nothing` was imported by nothing; the grounding check ran with an empty
answer because the drafter never passed one. Each had a passing fixture, and a passing fixture
plus an unreferenced function is indistinguishable from a passing fixture plus a wired-up one.
Every prior instance of D-023 ran and measured the wrong property; these never executed and
reported success. So the property asserted here is not "the check works" but "the check is
*reachable* from the code that builds and gates the set".

Two guards. The first is purely structural: every public function in the check module must
have at least one call site outside its own definition and outside the tests — a check no
production path can reach is an orphan. The second is the argument-shaped variant of the
same defect: a model check that is called, but called without the input that makes it
meaningful.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
CHECK_MODULE = ROOT / "evals" / "verify_items.py"
# Where a check may legitimately be reached from. Tests are excluded on purpose: a call from a
# test is exactly the false comfort this guard exists to remove.
PRODUCTION = [*sorted((ROOT / "evals").glob("*.py")), *sorted((ROOT / "scripts").glob("*.py"))]


def public_functions(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and not node.name.startswith("_")
    ]


def call_sites(name: str) -> list[str]:
    """Files (other than the definition line) that reference the function as a call."""
    pattern = re.compile(rf"(?<![\w.]){re.escape(name)}\(")
    found: list[str] = []
    for path in PRODUCTION:
        text = path.read_text(encoding="utf-8")
        hits = [m.start() for m in pattern.finditer(text)]
        if path == CHECK_MODULE:
            # discount the definition itself
            hits = [h for h in hits if not text[max(0, h - 4) : h].endswith("def ")]
        if hits:
            found.append(path.name)
    return found


class TestNoOrphanChecks:
    def test_every_public_check_has_a_production_call_site(self) -> None:
        orphans = {name: call_sites(name) for name in public_functions(CHECK_MODULE)}
        orphans = {name: sites for name, sites in orphans.items() if not sites}

        assert not orphans, (
            f"{sorted(orphans)} are defined in evals/verify_items.py and called from nowhere in "
            f"evals/ or scripts/. A check with a fixture and no call site reads exactly like a "
            f"working check (D-026). Wire it into classify_candidate or delete it."
        )

    def test_the_guard_itself_can_fail(self) -> None:
        """A guard that cannot fire is the defect it guards against (D-023)."""
        assert call_sites("a_function_nobody_defines_or_calls") == []

    def test_the_drafter_passes_the_answer_to_the_necessity_check(self) -> None:
        """The grounding check defaulted to True for a whole round because this was missing.

        Source inspection, deliberately: the property is "the argument is supplied", and a
        behavioural test would need a live model. `check_single_paper_sufficiency` grounds only
        when it is given an answer, so a call without one is a call to a check that cannot fire.
        """
        import inspect

        from evals.draft import draft_multi_hop

        source = inspect.getsource(draft_multi_hop)
        call = re.search(r"check_single_paper_sufficiency\((.*?)\)\n", source, re.DOTALL)

        assert call is not None
        assert "drafted.answer" in call.group(1), (
            "draft_multi_hop calls the necessity check without the drafted answer; grounding "
            "will default to True and never fire (D-026)"
        )
