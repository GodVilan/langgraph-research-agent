"""Prompt-injection defence on retrieved document text.

AUDIT §4.11 named this the largest gap in v2.1: retrieved chunk text was concatenated
straight into the ReAct scratchpad and the synthesis prompt with no delimiter, no
instruction/data separation, and no detector. The exposure was not theoretical, because
``fetch_arxiv`` downloads a PDF chosen by a *model-authored* query, extracts its text, and
feeds it back into the agent's own control loop.

Three layers, deliberately independent — each one catches things the others miss:

1. **Structural** (``neutralise``) — retrieved text can never close, forge, or nest the
   delimiter that wraps it. This holds regardless of what the detector knows about.
2. **Instructional** — the generate prompt states that passage content is data, names the
   delimiter, and says an instruction inside one is quoted material to report, not to obey.
3. **Detection** (``scan``) — pattern matching over known injection shapes, with every hit
   recorded as a ``GuardrailEvent`` so it reaches the trace.

Layer 1 is the one that does not depend on foreseeing the attack, so it is the one that
matters most. The detector is a defence in depth and a measurement instrument, not the
primary control — the honest position is that pattern matching cannot enumerate the space
of injections, and `tests/test_injection.py` reports where it fails as well as where it
succeeds.

**Trust tiers.** Corpus chunks come from a committed, checksummed corpus. ``arxiv`` and
``upload`` chunks arrive at runtime, chosen by a query the model wrote. The latter get the
strictest policy: anything a corpus chunk would merely be flagged for, a runtime chunk is
quarantined for.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

# ── Delimiters ────────────────────────────────────────────────────────────────

PASSAGE_OPEN: Final = "<passage"
PASSAGE_CLOSE: Final = "</passage>"

# Anything that could terminate, reopen, or impersonate the wrapper. Replaced with a
# visible marker rather than deleted, so the model can see that something was removed and
# the trace records what the original looked like.
_DELIMITER_RE = re.compile(r"</?\s*passage\b[^>]*>?", re.IGNORECASE)
_REDACTION = "[delimiter removed]"


class Category(StrEnum):
    INSTRUCTION_OVERRIDE = "instruction_override"
    ROLE_HIJACK = "role_hijack"
    EXFILTRATION = "exfiltration"
    DELIMITER_ESCAPE = "delimiter_escape"
    TOOL_INVOCATION = "tool_invocation"
    OUTPUT_CONTROL = "output_control"
    ENCODING_EVASION = "encoding_evasion"


class Severity(StrEnum):
    WARN = "warn"
    BLOCK = "block"


@dataclass(frozen=True)
class Detection:
    category: Category
    severity: Severity
    pattern: str
    excerpt: str

    def describe(self) -> str:
        return f"{self.category.value}/{self.pattern}: {self.excerpt!r}"


@dataclass(frozen=True)
class Rule:
    name: str
    category: Category
    severity: Severity
    regex: re.Pattern[str]


# Sentence boundary that survives whitespace collapsing. `scan()` flattens newlines before
# matching, so `^` under re.MULTILINE degenerates to "position 0 of the chunk" — an anchored
# rule would then only ever fire on a payload that happens to start the chunk, which is a
# corpus artifact rather than a property of real documents. Every anchored rule uses this
# instead, and re.MULTILINE is deliberately NOT set so nothing silently depends on it.
SENTENCE_START = r"(?:^|(?<=[.!?:;)\]\"'\u00bb\u201d])\s)"


def _r(name: str, category: Category, severity: Severity, pattern: str) -> Rule:
    return Rule(name, category, severity, re.compile(pattern, re.IGNORECASE))


# Severity is assigned from corpus evidence, not intuition (docs/SCREEN.md).
#
# This corpus is 150 papers about LLMs and agents, so legitimate chunks quote system
# prompts, tool calls, and instruction text constantly. Rules keyed on *prompt-like
# language* therefore cannot be high-precision here, and are WARN: they surface in the
# trace, and they block only under `strict`, which applies to runtime-fetched content —
# the actual threat surface (AUDIT §4.11).
#
# Rules keyed on *structural artifacts* — a forged delimiter, an invisible character, a
# markup comment, our own tool-call protocol — stay BLOCK, because those do not occur in
# legitimately extracted paper prose. The first screen confirmed the split: before this
# retune, 175 of 5,401 corpus chunks (3.24%) would have been quarantined, and inspection
# of every firing rule found no true positive among them.
RULES: Final[tuple[Rule, ...]] = (
    # ── instruction override ──────────────────────────────────────────────────
    _r(
        "ignore_previous",
        Category.INSTRUCTION_OVERRIDE,
        Severity.BLOCK,
        r"\b(ignore|disregard|forget|override)\b[^.\n]{0,40}\b"
        r"(previous|prior|earlier|above|all)\b[^.\n]{0,30}\b"
        r"(instruction|prompt|direction|rule|command|context)",
    ),  # 0 corpus hits — specific enough to stay BLOCK
    _r(
        "new_instructions",
        Category.INSTRUCTION_OVERRIDE,
        Severity.BLOCK,
        # `task` and `objective` are removed: "…to New Tasks: A Relational Approach" is a
        # paper title and accounted for 28 of this rule's 29 corpus hits. Requiring a
        # sentence boundary and a colon keeps the header-style injection shape.
        SENTENCE_START + r"(new|updated|revised|ignore\s+the\s+above)\s+"
        r"(instruction|directive)s?\b\s*[:\-]",
    ),
    _r(
        "imperative_you_must",
        Category.INSTRUCTION_OVERRIDE,
        Severity.WARN,
        r"\byou\s+(must|shall|should|need to|are required to|are instructed to)\b"
        r"[^.\n]{0,40}\b(now|instead|immediately|always)\b",
    ),
    _r(
        "stop_and_do",
        Category.INSTRUCTION_OVERRIDE,
        Severity.BLOCK,
        r"\bstop\b[^.\n]{0,20}\b(what|everything|following)\b[^.\n]{0,30}\band\b",
    ),  # 0 corpus hits
    # ── role hijack ───────────────────────────────────────────────────────────
    _r(
        "system_role_claim",
        Category.ROLE_HIJACK,
        Severity.WARN,
        # D-2: was `^…` under re.MULTILINE, which after whitespace collapsing could only
        # match position 0 — the rule went silent on any payload placed mid-chunk, and the
        # corpus case passed only because its payload started the chunk.
        # WARN, not BLOCK: papers about agents quote "System: You are a cautious…" verbatim.
        SENTENCE_START + r"(system|assistant)\s*[:\]]\s*\S",
    ),
    _r(
        "markup_comment_directive",
        Category.ROLE_HIJACK,
        Severity.BLOCK,
        # An HTML/XML comment in text extracted from a PDF is already anomalous; one that
        # addresses a role or issues a directive is not prose. Found by probing.
        r"<!--[^>]{0,200}?\b(system|assistant|user|ignore|instruction|you must)\b",
    ),
    _r(
        "fake_turn_marker",
        Category.ROLE_HIJACK,
        Severity.WARN,
        # Narrowed to markers that carry a *role*. Bare `<|endoftext|>` accounted for 8 of
        # this rule's 9 original corpus hits and is ordinary content in tokenizer papers.
        # The narrowed form still hit 3 chunks, all of them papers quoting real
        # chat-template transcripts, so it is WARN rather than BLOCK: a paper about LLM
        # evaluation containing `<|im_start|>user` is evidence, not an attack. It still
        # blocks under `strict` on runtime-fetched content, which is the threat surface.
        r"(<\|(im_start|im_end|start_header_id|end_header_id)\|>"
        r"|<\|[^|>]{0,16}\|>\s*(system|assistant|user)\b"
        r"|\[/?INST\]"
        r"|###\s*(system|assistant|instruction))",
    ),
    _r(
        "you_are_now",
        Category.ROLE_HIJACK,
        Severity.WARN,  # 12 corpus hits, all papers quoting agent system prompts
        r"\byou\s+are\s+(now\s+)?(a|an|the)\s+[\w\s]{0,30}"
        r"(assistant|model|agent|bot|dan|admin|developer)\b",
    ),
    _r(
        "pretend_roleplay",
        Category.ROLE_HIJACK,
        Severity.WARN,
        r"\b(pretend|act as if|roleplay as|behave as)\b[^.\n]{0,30}\byou\b",
    ),
    # ── exfiltration ──────────────────────────────────────────────────────────
    _r(
        "reveal_prompt",
        Category.EXFILTRATION,
        Severity.BLOCK,
        # `token` alone matched "output token", "show that tokens…" and similar across 39
        # corpus chunks — ubiquitous ML vocabulary. Credentials must now be named as such,
        # and prompt/instruction targets require a second-person possessive, which is what
        # separates "we reveal that…" from "reveal your system prompt".
        r"\b(reveal|print|output|repeat|show|disclose|echo|dump)\b[^.\n]{0,30}\b"
        r"(your\s+(system\s+)?(prompt|instructions|configuration)"
        r"|the\s+system\s+prompt"
        r"|api[\s_-]?key|secret\s+key|credential|password"
        r"|(access|auth|bearer|api)[\s_-]token)",
    ),
    _r(
        "send_data_out",
        Category.EXFILTRATION,
        Severity.BLOCK,
        r"\b(send|post|upload|transmit|forward|exfiltrate)\b[^.\n]{0,40}"
        r"(https?://|www\.|@[\w.-]+\.\w+)",
    ),
    _r(
        "markdown_image_beacon",
        Category.EXFILTRATION,
        Severity.BLOCK,
        r"!\[[^\]]*\]\(\s*https?://[^)]*[?&][^)]*=",
    ),
    # ── delimiter escape ──────────────────────────────────────────────────────
    _r("passage_delimiter", Category.DELIMITER_ESCAPE, Severity.BLOCK, r"</?\s*passage\b"),
    _r(
        "fence_break",
        Category.DELIMITER_ESCAPE,
        Severity.WARN,
        r"```\s*(system|assistant|instruction)",
    ),
    # ── tool invocation ───────────────────────────────────────────────────────
    _r(
        "tool_call_json",
        Category.TOOL_INVOCATION,
        Severity.BLOCK,
        r'"(action|tool|tool_name|function)"\s*:\s*"',
    ),
    _r("react_protocol", Category.TOOL_INVOCATION, Severity.BLOCK, r'"(thought|action_input)"\s*:'),
    _r(
        "call_tool_imperative",
        Category.TOOL_INVOCATION,
        Severity.BLOCK,
        # Generic `tool`/`function` targets matched "call to a library function" and
        # "Call the `run_code` tool" across 11 corpus chunks. Only this agent's own tool
        # names remain — naming them is the attack, discussing tools in general is not.
        r"\b(call|invoke|execute|run)\b[^.\n]{0,20}\b"
        r"(fetch_arxiv|search_corpus|keyword_search)\b",
    ),
    _r(
        "conditional_ai_reader",
        Category.ROLE_HIJACK,
        Severity.BLOCK,
        r"\bif\s+you\s+(are|were)\b[^.\n]{0,30}\b"
        r"(ai|a\.i\.|language model|llm|assistant|agent|bot|summaris|summariz|reading)",
    ),
    # ── output control ────────────────────────────────────────────────────────
    _r(
        "answer_verbatim",
        Category.OUTPUT_CONTROL,
        Severity.WARN,
        # 51 corpus hits, the largest single source of false positives: papers about
        # prompting quote instructions like "Output only N/E/S/W", and "if and only if"
        # matched too. `only` now has to attach to the verb, and the mathematical
        # "if and only if" is excluded outright.
        # `say` is dropped: "…(say Ck') if and only if…" is mathematical prose, not an
        # instruction, and it survived the if/and lookbehinds by starting the match at
        # "say". The residual hits are papers quoting their own prompts, which is exactly
        # what WARN is for on this corpus.
        r"(?<!if\s)(?<!and\s)\b(answer|reply|respond|output)\b[^.\n]{0,15}?\b"
        r"(only|exactly|verbatim|nothing else)\b",
    ),
    _r(
        "cite_only_this",
        Category.OUTPUT_CONTROL,
        Severity.WARN,
        r"\b(cite|reference|recommend)\b[^.\n]{0,20}\bonly\b[^.\n]{0,25}\b(this|our)\b",
    ),
)

# Letter-spacing evasion: "I g n o r e   a l l".
#
# D-3: as a bare shape rule this fired on 33 corpus chunks, every one of them legitimate —
# maths variable runs ("Let x y z u v w t denote…"), letter-spaced headings
# ("A R T I C L E   I N F O"), and binary/label sequences ("0 1 0 1 1 0 0 1"). Those are
# routine in PDF-extracted text, not edge cases.
#
# The fix is to test the *reconstruction* rather than the shape: strip the spaces and see
# whether the result trips a real rule. That is precisely the attack shape ("I g n o r e
# a l l p r e v i o u s" reconstructs to an override) and precisely not the false-positive
# shape (a variable run reconstructs to nothing meaningful). A rule keyed on appearance
# blocks papers; a rule keyed on what the text becomes blocks attacks.
_SPACED_OUT_RE = re.compile(r"(?:\b\w\s){6,}\w\b")

# Characters with no business in extracted paper text: bidi overrides and zero-width
# joiners are used to hide an instruction from a human reviewer while leaving it visible to
# the model.
INVISIBLE_RE = re.compile(r"[​-‏‪-‮⁠-⁤﻿]")


def scan(text: str, strict: bool = False) -> list[Detection]:
    """Return every rule that fires on ``text``.

    ``strict`` promotes WARN to BLOCK. It is set for runtime-fetched content (arXiv,
    uploads), where the provenance chain runs through a model-authored query.
    """
    detections: list[Detection] = []

    if INVISIBLE_RE.search(text):
        found = INVISIBLE_RE.findall(text)
        detections.append(
            Detection(
                Category.ENCODING_EVASION,
                Severity.BLOCK,
                "invisible_characters",
                f"{len(found)} invisible char(s), first U+{ord(found[0]):04X}",
            )
        )

    # Normalise before matching so that compatibility and homoglyph forms of a word
    # (fullwidth Latin, mathematical alphanumerics) do not walk past a plain-ASCII regex.
    normalised = unicodedata.normalize("NFKC", text)
    # Collapse whitespace runs, including newlines, so that splitting a phrase across lines
    # does not defeat patterns whose gaps exclude newlines. Found by probing, not by design:
    # "Ignore all\nprevious\ninstructions" evaded every rule before this line existed.
    #
    # D-2: this is a detector-wide behaviour change, not a local fix. It removes every
    # newline before matching, so `^`/`$` under re.MULTILINE would degenerate to the chunk
    # boundaries. That flag is deliberately not set; anchored rules use SENTENCE_START.
    normalised = re.sub(r"\s+", " ", normalised)

    # Runs on the collapsed text: a letter-spaced phrase is usually separated by *double*
    # spaces between words ("I g n o r e   a l l"), which splits it into fragments too short
    # to reconstruct unless the runs are joined first.
    detections.extend(_scan_letter_spacing(normalised))

    for rule in RULES:
        match = rule.regex.search(normalised)
        if match is None:
            continue
        severity = Severity.BLOCK if strict and rule.severity is Severity.WARN else rule.severity
        excerpt = normalised[max(0, match.start() - 20) : match.end() + 20].replace("\n", " ")
        detections.append(Detection(rule.category, severity, rule.name, excerpt.strip()))

    return detections


# De-spacing destroys word boundaries by construction, so the reconstruction is tested
# against bare keywords rather than the boundary-anchored RULES above.
_SPACED_KEYWORDS: Final[tuple[str, ...]] = (
    "ignoreall",
    "ignoreprevious",
    "ignorethe",
    "disregard",
    "override",
    "previousinstruction",
    "newinstruction",
    "systemprompt",
    "youarenow",
    "revealyour",
    "apikey",
    "forgetall",
    "forgeteverything",
)


def _scan_letter_spacing(text: str) -> list[Detection]:
    """Flag a letter-spaced run only when de-spacing it yields an injection keyword.

    Keying on the *reconstruction* rather than the shape is what separates the attack
    ("I g n o r e   a l l" -> "ignoreall") from the false positives that shape matching
    produced on the real corpus ("Let x y z u v w t denote…" -> "xyzuvwt").
    """
    for match in _SPACED_OUT_RE.finditer(text):
        run = match.group()
        reconstructed = re.sub(r"\s+", "", run).lower()
        if len(reconstructed) < 8:
            continue
        hit = next((k for k in _SPACED_KEYWORDS if k in reconstructed), None)
        if hit is not None:
            return [
                Detection(
                    Category.ENCODING_EVASION,
                    Severity.BLOCK,
                    "letter_spacing",
                    f"{run[:40]!r} -> {reconstructed[:40]!r} (keyword {hit!r})",
                )
            ]
    return []


def neutralise(text: str) -> str:
    """Make text structurally incapable of escaping its wrapper.

    This is the layer that does not depend on recognising the attack: whatever the content
    says, it cannot close the ``<passage>`` block it sits in, because every delimiter-shaped
    token is replaced before rendering. Invisible characters are stripped for the same
    reason — they exist to make the rendered text differ from what a human reviewing the
    corpus would see.
    """
    cleaned = INVISIBLE_RE.sub("", text)
    return _DELIMITER_RE.sub(_REDACTION, cleaned)


def worst_severity(detections: list[Detection]) -> Severity | None:
    if any(d.severity is Severity.BLOCK for d in detections):
        return Severity.BLOCK
    if detections:
        return Severity.WARN
    return None


def is_untrusted_source(source: str, retriever: str) -> bool:
    """Runtime-fetched content gets the strict policy.

    Corpus chunks come from a committed, checksummed artifact. Anything else arrived during
    the request, selected by a query the model wrote (AUDIT §4.11).
    """
    return source != "corpus" or retriever == "arxiv"
