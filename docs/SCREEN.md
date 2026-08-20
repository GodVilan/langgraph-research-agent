# CORPUS SCREEN — injection detector against the real corpus

Regenerate:

```bash
make screen-corpus
```

Pure regex over the 5,401 committed chunks. No LLM calls, no API spend, a few seconds of
local compute.

This exists because the guardrail's only false-positive number was 0/8 on a benign set the
same author wrote. That measures what I thought to write down. The corpus measures what the
rules actually do to real PDF-extracted text, and the two answers were not close.

---

## Result

| | Before retune | After retune |
|---|---:|---:|
| Chunks screened | 5,401 | 5,401 |
| Chunks with ≥1 detection | 176 (3.26%) | **60 (1.11%)** |
| **Would be quarantined, corpus policy** | **175 (3.24%)** | **0 (0.00%)** |
| Would be quarantined, `strict` policy | 176 (3.26%) | 60 (1.11%) |

Detection on the adversarial corpus is unchanged by the retune: 25/25 targeted cases still
detected, and **25/25 quarantined under `strict`** — the policy that applies to
runtime-fetched content, which is the actual threat surface (AUDIT §4.11). Under the corpus
policy 19/25 quarantine and 6 warn, which is the deliberate trade described below.

---

## What the first screen found

Every one of the 175 chunks that would have been quarantined was legitimate academic text.
Not "mostly" — inspection of the top excerpts for all nine firing rules turned up no true
positive.

| Rule | Hits | What it was actually matching |
|---|---:|---|
| `answer_verbatim` | 51 | Papers quoting their own prompts: `"Output only N/E/S/W"`, `"Output valid JSON only"`. Also `"if and only if Ck′ = arg min"` — the match started at `say` in `(say Ck′)`. |
| `reveal_prompt` | 39 | The bare word `token`. `"output tokens"`, `"show that … tokens"`, `"a 1024 output-token cap"`. Ubiquitous ML vocabulary. |
| `letter_spacing` | 33 | `"Let x y z u v w t denote"`, `"A R T I C L E   I N F O"`, `"0 1 0 1 1 0 0 1"`. Maths variable runs and letter-spaced headings. |
| `new_instructions` | 29 | 28 of them one paper title: `"Learning to Extrapolate to New Tasks: A Relational Approach"`. |
| `you_are_now` | 12 | Papers quoting agent system prompts: `"the agent's role as: 'You are an incident response agent'"`. |
| `call_tool_imperative` | 11 | `"call to a library function"`, `"Call the run_code tool"`. |
| `fake_turn_marker` | 9 | 8 were bare `<|endoftext|>` in tokenizer papers. |
| `system_role_claim` | 2 | Papers quoting `"System: You are a cautious…"`. |
| `imperative_you_must` | 1 | `"react immediately. You must always include evidence"`. |

### The systemic cause

This corpus is 150 papers about LLMs and agents. They quote system prompts, tool calls, and
instruction text constantly, because that is their subject matter. **A detector keyed on
prompt-like language cannot be high-precision on a corpus of papers about prompting.**

That is not a tuning problem, it is a category problem, and it drove the severity policy
below rather than a round of pattern-fiddling.

---

## The retune

### Severity is now assigned by what a rule keys on

| Keys on | Severity | Why |
|---|---|---|
| **Structural artifacts** — forged delimiter, invisible characters, markup comment, our own tool-call protocol, letter-spacing that reconstructs to an injection keyword | `BLOCK` | These do not occur in legitimately extracted paper prose. Corpus evidence: 0 hits. |
| **Prompt-like language** — role claims, "you are now", "answer only", turn markers | `WARN` | Papers about LLMs contain these legitimately. They surface in the trace and block only under `strict`, which applies to runtime-fetched content. |

A `WARN` on corpus text changes nothing: the chunk still reaches the model, the event still
reaches the trace. A `BLOCK` silently withholds evidence, and on this corpus it was
withholding exactly the papers most relevant to questions about agents and prompting.

### Pattern fixes, each traceable to a specific false positive

| Rule | Change |
|---|---|
| `reveal_prompt` | `token` alone removed. Credentials must be named (`api key`, `access token`, `secret key`); prompt targets need a second-person possessive — which separates *"we reveal that…"* from *"reveal your system prompt"*. |
| `new_instructions` | `task` and `objective` dropped, sentence boundary and colon required. Kills the paper-title match. |
| `call_tool_imperative` | Generic `tool`/`function` targets removed; only this agent's own tool names remain. Naming them is the attack; discussing tools is not. |
| `fake_turn_marker` | Narrowed to role-bearing markers, so bare `<\|endoftext\|>` no longer fires. Then downgraded to `WARN`, because the 3 surviving hits were papers quoting real chat transcripts. |
| `answer_verbatim` | `say` dropped (killed the `if and only if` match), lookbehinds kept, downgraded to `WARN`. |
| `system_role_claim` | Anchor fixed (see D-2 below) and downgraded to `WARN`. |
| `letter_spacing` | Rewritten to key on the reconstruction, not the shape (see D-3 below). |

---

## D-2 — the whitespace collapse had silently disabled an anchor

`scan()` collapses all whitespace before matching, which was the right fix for a phrase
split across newlines. But `system_role_claim` was anchored `^` under `re.MULTILINE`, and
after the collapse there are no newlines left — so `^` could only ever match position 0 of
the whole chunk.

The rule was live-tested and green. It passed only because its corpus payload happened to
start at character 0. Move the identical text one sentence later — which is how it would
appear in a real paper — and the rule went silent.

**Neither the 33-case adversarial corpus nor the 12 fresh evasion probes could see this**,
because every anchored payload had been written at position 0. That is a corpus artifact,
not a property of documents.

Fixed by:

- replacing `^` with `SENTENCE_START`, an explicit boundary that survives the collapse;
- **not** setting `re.MULTILINE` anywhere, so nothing silently depends on newlines that the
  normalisation step has already removed;
- adding a mid-chunk variant of every anchored payload to the corpus permanently
  (`system_prefix_mid_chunk`, `new_directive_mid_chunk`, `assistant_marker_mid_chunk`).

The general lesson is recorded in `DECISIONS.md`: collapsing whitespace is a detector-wide
behaviour change, not a local fix, and any future change to the normalisation step needs the
same audit of every positional anchor.

---

## D-3 — letter-spacing keyed on appearance, not meaning

`(?:\b\w\s){6,}\w\b` fired on 33 corpus chunks, all legitimate. Maths variable enumerations
and letter-spaced headings are routine in PDF extraction, and the rule was `BLOCK` with no
`strict` gate, so it quarantined outright.

Rather than guessing at a threshold, the rule now tests the **reconstruction**: strip the
spaces and check whether the result contains an injection keyword.

- `"I g n o r e   a l l"` → `ignoreall` → fires. That is the attack shape.
- `"Let x y z u v w t"` → `xyzuvwt` → nothing. That is the false-positive shape.

A rule keyed on appearance blocks papers; a rule keyed on what the text becomes blocks
attacks. Corpus hits went 33 → 0 while both spaced attack variants are still caught.

One subtlety worth recording: the reconstruction is matched against bare keywords rather
than the main `RULES`, because de-spacing destroys the word boundaries every rule depends
on. And the scan runs on the *collapsed* text, because a spaced phrase is usually separated
by double spaces between words, which otherwise splits it into fragments too short to
reconstruct.

All four false-positive strings are now permanent `BENIGN` regression cases, along with the
`A R T I C L E   I N F O` and binary-sequence shapes.

---

## What this screen does not tell you

- **It measures precision, not recall.** 0% quarantine on the corpus is only good news
  because the corpus is believed clean. If it contained a real injection, this screen would
  not distinguish "no false positives" from "missed it".
- **It is one corpus.** 150 cs.LG papers from a single day. A corpus with a different
  subject mix would produce a different false-positive profile — a corpus with *no* papers
  about prompting would have made the original rules look fine.
- **The 60 remaining WARNs are still almost entirely false positives.** They are tolerated
  because a warning changes no behaviour, but they will be noise in Phase 3's traces, and
  the guardrail-trigger metric needs to separate `warn` from `block` or it will look alarming
  for no reason.
