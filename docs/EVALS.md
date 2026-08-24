# EVALS — construction rules and hazards

How the Phase 4 eval set is built, and the ways of building it that produce items which
look correct and are not. Every hazard below was hit here, not imagined.

Related: [`BUDGET.md`](./BUDGET.md) (the $5 judge ceiling), [`DECISIONS.md`](./DECISIONS.md),
[`AUDIT.md`](./AUDIT.md) §5 (why v2.1's benchmark is unusable, and the standard this set has
to meet instead).

---

## The strata

| Stratum | Target | What it measures | Supply |
|---|---:|---|---|
| `single_paper_factual` | 55 | Retrieval + faithful extraction | 142/150 papers eligible |
| `multi_hop` | 20 | Synthesis across papers | 314 candidate pairs, **before** the necessity check |
| `unanswerable_topic` | 4 | **Scope refusal** — "the corpus does not contain this" | Hard-capped at 4 |
| `unanswerable_attribute` | 11 | **Attribute refusal** — "this paper does not report that" | 979 verified candidates |
| `ambiguous` | 10 | Clarification rather than a confident guess | 22 RL / 21 alignment / 14 diffusion papers |

Regenerate the supply figures with `make corpus-diversity`.

### The two unanswerable strata are never merged

They test different capabilities. A system can be good at "I have never heard of curriculum
learning" and bad at "this paper exists, and it does not report that number" — the second is
much harder, because retrieval returns the anchor paper's own chunks at high similarity and
everything about the context says *answer confidently*.

Reporting them together as "refusal accuracy on n=15" would average two skills and hide
which one failed. They are separate types in the schema, they carry separate
`reporting_group` values, and **their accuracies are always reported as n=4 and n=11
separately.** Never a blended figure.

The n=4 is a real shortfall against the original target of 15 and is reported as one.

---

## Hazard 1 — the false-absence hazard

**The rule: an item is only unanswerable if it is unanswerable everywhere the agent is
allowed to look.**

This is the hazard that nearly produced a broken stratum, and it has two forms. Both are the
same mistake — checking absence in a smaller place than the agent searches.

### 1a. Screening against abstracts instead of full text

Screening 35 candidate topics against the 150 abstracts marks 28 of them absent. Screening
the same 35 against all 5,401 chunks leaves **4**.

**24 of the 28 apparent absences are discussed in the body of the corpus** — an 86%
false-absence rate. Mixture-of-experts, active learning, knowledge graphs, meta-learning,
OCR, SVMs, random forests and seventeen others all look absent from the abstracts and are
all discussed in related-work sections.

The cause is structural, not incidental: 13.4M characters of related work name nearly every
term in machine learning at least once in passing. Any corpus of research papers will behave
this way.

**Why this is worse than an ordinary bug.** An item built this way is not merely wrong, it is
*backwards*, and it fails silently in the direction that flatters the system. The expected
answer is a refusal; the corpus does contain the material; a correct, well-grounded answer
is scored as a failure to refuse. The eval would report a refusal-accuracy number that
punishes the agent for being right, and nothing in the run would look unusual.

### 1b. Screening against the anchor paper instead of the whole corpus

For `unanswerable_attribute` items, "paper X never mentions SQuAD" is not enough. If paper
Y's related work reports X's SQuAD score, the corpus answers the question, and the item is
backwards in exactly the same way.

`evals/absence.py::verify_attribute_absent` therefore checks both: the term is absent from
the anchor paper, **and** no other chunk mentions the term alongside a handle for the anchor
paper (its arXiv id, or a distinctive method name from its title).

**How much this eliminates here: 0 of 979 candidate pairs.** That is not a reason to drop the
check, and the reason it is zero is worth stating — all 150 papers were published on the same
afternoon, so no paper *can* cite another's results. The check costs nothing, it is the
discipline whose absence cost the topic stratum, and it becomes load-bearing the moment the
corpus spans more than one day.

A caveat that is a real limit rather than a formality: only **45 of 150** papers have a
method name distinctive enough to match on. For the other 105 the check rests on the arXiv id
alone, which is weaker. Substring matching is not an option — matching `Gram` inside
"n-gram", "program" and "diagram" produced two false eliminations before word boundaries were
added.

---

## Hazard 2 — topical relatedness is not multi-hop necessity

`make corpus-diversity` reports 314 paper pairs sharing four or more distinctive terms. That
is 314 pairs worth *trying*, not 314 multi-hop items. Shared vocabulary means two papers are
about related things, which is a precondition for a multi-hop question and no evidence that
any particular question needs both papers.

**Both conditions are required**, and `evals/multihop.py` checks both by putting the question
to each paper alone and then to the papers together:

1. no single source paper answers it, and
2. the source papers *together* do answer it.

Condition 2 was added after a live run, and the case that forced it is instructive. A
question about federated learning was put to two papers about alignment auditing and Bayesian
networks. No single paper could answer it, so the first version of the check passed it as
"genuinely multi-hop". Neither could the pair. **It was not a hard item, it was a broken
one** — and it would have entered the set as the hardest stratum and been scored as a
hallucination no matter what the agent said.

Verified behaviour on the real corpus, on a genuine pair (two federated-learning papers,
`2605.30075` and `2605.30123`):

| Question | Single-paper | Joint | Verdict |
|---|---|---|---|
| "How do these two differ in what they do to client updates before aggregation?" | neither | yes | **multi-hop** |
| "What is ZNE-guided correction and why is it needed?" | `2605.30075` yes | yes | rejected — single-paper |
| federated question put to two unrelated papers | neither | **no** | rejected — nothing answers it |

If the necessity check eliminates a large share of the 20, that is reported as a shortfall.
An item that only looks multi-hop inflates the hardest stratum with easier items and makes
the headline number better than the system is.

### The necessity check is circular, and stays that way

`evals/multihop.py` decides whether a question needs two papers by asking
`gemini-3.5-flash-lite` — **the same model family that generates the agent's answers.**
Multi-hop difficulty is therefore calibrated to the reading ability of the system under
test. If that model is unusually good at synthesising from one paper, questions it can
answer alone are culled, and the surviving stratum is *systematically easier* for the agent
than a human-labelled stratum would be. The bias runs in the flattering direction.

This is not fixed. Fixing it means either a second model family for necessity checking —
which is the judge budget, spent on item construction instead of judging (D-001) — or
hand-labelling all twenty, which is most of the human verification budget for one stratum.
Neither is worth it at n=20.

What is done instead: the limitation is stated here, the necessity check's model and
snapshot are recorded in each item's provenance, and **10 of the 25 hand-verified items are
multi-hop** — a deliberate over-weighting of the stratum that has no other validation. The
human-vs-machine agreement rate on those 10 is the only independent evidence about this
check, which is why the verification CLI takes the human decision before revealing the
machine verdict.

### Measured cull rate, and what it actually measured

A first pass drafted 8 questions from the 8 most strongly-related paper pairs and put them
through the necessity check. **7 of 8 were culled (87.5%)** — and every one for the same
reason: `neither-nor-joint`, meaning the two papers *together* do not answer the question.
None was culled for being single-paper.

That number is not the corpus's multi-hop capacity. It is a measurement of the drafting
prompt, and the drafted questions say why:

> "How can multi-key homomorphic encryption for secure client aggregation be integrated
> into…"
> "How can the trajectory-based dynamic weighting scheme from the second study be
> integrated into…"

These are **research proposals, not questions**. Asked to write something requiring both
papers, the model proposed combining them — which no paper answers, because nobody has done
it. They read as sophisticated multi-hop questions and are unanswerable by construction.

Two causes, both mine:

1. The drafter was given **abstracts**, not papers — violating this document's own first
   construction rule. With 1,200 characters of abstract it cannot know what either paper
   actually reports, so it writes about what they are *about*.
2. The word "synthesis" in the drafting prompt invites invention. The prompt must require
   that both papers *contain* the facts needed, and explicitly forbid asking how one method
   could be applied to the other's setting.

**The check did its job.** Seven items that would have entered the hardest stratum as
unanswerable-by-construction were caught before a human ever saw them, and the failure mode
they share is legible because the rejection reason distinguishes `neither-nor-joint` from
`single-paper`. A cull rate reported without that breakdown would have looked like corpus
scarcity and prompted the wrong fix — padding the stratum from weaker pairs.

**The fixed prompt, piloted on 5 pairs before committing to the set:**

| | first pass | after the fix |
|---|---:|---:|
| Drafted | 8 | 5 |
| Cull rate | **87.5%** | **20%** |
| `neither_nor_joint` | 7 | **0** |
| `single_paper` | 0 | 1 |
| `banned_phrasing` | n/a | 0 |

The shift from `neither_nor_joint` to `single_paper` is the signal that the prompt is
fixed. Zero unanswerable questions; the one cull is a pair too closely related to need both
papers, which is ordinary difficulty calibration rather than a construction fault. Piloting
five cost about five minutes at 15 RPM instead of forty on a full run.

The kept questions have the right shape — *"What number of communication rounds do the two
papers report for evaluating their models on CIFAR…"*, *"Which of the two evaluates model
representations on a grid maze environment, and which evaluates…"* — asking what the papers
report rather than what could be built from them.

`ConstructionReport.by_reason` is a permanent field, never pooled, and
`ConstructionReport.diagnosis()` names which of the three failure modes dominates.

---

## Hazard 3 — one paper measured several times

If a paper anchors an `unanswerable_attribute` item *and* sources a `multi_hop` item, those
two measurements are correlated. A quirk in that paper — an unusually formatted results
table, idiosyncratic phrasing, a section the chunker split badly — then surfaces as two
findings that look independent and are not. With five strata over 150 papers it is easy to
let this happen without noticing.

**It is avoidable here, so it is enforced rather than accepted:**

| | |
|---|---:|
| Multi-hop candidate pairs | 6,456 |
| Pairs containing no attribute anchor | **5,598** |
| Distinct papers in the 60 cleanest pairs | 66 |
| Of those, overlapping the 11 attribute anchors | **0** |
| Factual-eligible papers after excluding both | 71 |

The budget is 55 factual + ~40 multi-hop papers + 11 anchors = 106 paper-slots across 150
papers, so disjoint strata fit with room to spare.

`EvalSet.cross_stratum_overlap()` reports any paper shared between two strata, and
`EvalItem.shared_papers` records it per item. Both exist for the case where disjointness
stops being achievable — a smaller corpus, or more strata — so the overlap would be
*recorded* rather than engineered around or silently absorbed.

---

## Hazard 4 — one item measured eleven times

Eleven `unanswerable_attribute` items all shaped "what did X report on benchmark Y" is a
single item with eleven anchors. It would produce a confident-looking n=11 that varies in
almost nothing.

`AbsenceShape` records what kind of fact is missing, and the balance is checked when the set
is frozen. The shapes are **benchmark, dataset, ablation, baseline, hyperparameter,
compute** — a missing accuracy number, a corpus never trained on, an ablation never run, a
baseline never compared against, a hyperparameter never stated, and a hardware or cost figure
never given.

---

## Construction rules

These come from the Phase 4 plan and are enforced in code where enforcement is possible.

* Generate from the **paper**, not the chunk.
* Forbid verbatim reuse of distinctive multi-word phrases, model names, and dataset names
  where a paraphrase exists. Automated token-overlap check against the gold chunk, rejecting
  above a documented threshold (`Verification.lexical_overlap_with_gold`).
* Draft with **Gemini**, never OpenAI. OpenAI is the judge and only the judge (D-001), and
  item construction is not judging. The multi-hop necessity check also runs on Gemini.
* Freeze with a schema version, a payload checksum, and a `generated_by` recording model,
  snapshot, prompt version, and date.
* Assert `data/CORPUS.sha256` at eval startup and record it in every baseline JSON. This is
  the record whose absence made v2.1's benchmark unusable when its corpus was swapped.
* 25 items verified by hand, weighted toward the strata that rest entirely on automated
  checks rather than toward the set's own proportions: **10 multi-hop, 8
  unanswerable-attribute, 5 single-paper factual, 2 ambiguous.** Single-paper factual has a
  gold chunk a human can read in seconds and does not need the scrutiny; multi-hop and
  attribute have nothing else.
* 25 items verified by hand. Generator-vs-human disagreement is **reported, not resolved
  silently** — rejected items stay in the file carrying their reason.

---

## Verifying by hand

```bash
make verify-evals FILE=evals/datasets/draft.json     # verify interactively
python -m evals.verify_cli evals/datasets/draft.json --report    # progress only
python -m evals.verify_cli evals/datasets/draft.json --stratum multi_hop
```

**The human decides first. Automated verdicts are revealed only afterwards.**

This ordering is the point of the tool, not a detail of it. Displaying `necessity: PASS`
before the decision makes the label dependent on the check, and the agreement rate then
measures agreement-with-the-machine rather than generator-vs-human disagreement. Multi-hop
and unanswerable-attribute have no validation *other* than that number — every other check
on them is automated — so it is worthless unless arrived at independently.

So the screen shows the question, the gold chunks or the claimed absent term, and the
provenance; the decision and its reason are taken; and only then do the machine verdicts
appear, alongside whether the two agree. Both labels are stored (`human_accepted` and
`machine_checks`), never merged, and each disagreement is attributable to a **named** check
(`topic_absence`, `attribute_absence`, `gold_chunks_exist`, `multi_hop_necessity`) rather
than to "the pipeline". `--report` prints agreement per stratum, never pooled.

Disagreement is recorded, not resolved. A human acceptance over a failed check means the
check is over-strict; a human rejection over passing checks means every check missed
something. Both are findings.

Rejection is one keystroke and takes a reason. Progress is written after every decision, so a
crash at item 20 of 25 does not cost the first 19. The stratum is deliberately not editable —
changing it would move an item between reporting groups and blur the two unanswerable
sub-strata, so a mis-stratified item is rejected and redrafted instead.
