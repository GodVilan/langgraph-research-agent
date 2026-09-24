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

### `unanswerable_topic` is a case study at n=2 — and effectively n=1 clean (D-031)

Paraphrase-aware screening across all 5,401 chunks leaves **two** genuinely absent topics:
AlphaFold and click-through rate. Sixteen candidates were probed; fourteen are discussed
under some surface form the literal string missed — the corpus never writes "curriculum
learning" but does order examples "easy to hard", never writes "capsule networks" but does
describe "dynamic routing".

**Two items are reported as a pass/fail case study with `n=2` stated inline, never as an
accuracy.** Two items yield only 0%, 50% or 100%, and any of those printed as a rate is
noise with a decimal point attached. It sits beside `unanswerable_attribute` at n≈9–11,
which *is* reported as a genuine rate, and the two are never pooled.

The stratum stays at two rather than being dropped, because scope refusal — "the corpus does
not contain this" — is a capability nothing else in the set measures. Dropping it would
delete a capability from the report, not simplify it.

**After scoring, it is effectively n=1 clean.** `ut-003` was built on the absent term
*AlphaFold* and asks about *protein structure prediction results*; the corpus has OmegaFold
doing exactly that. The screen verified the seed term, the question generalised above it, and
**the agent was right and the item was wrong** — it named OmegaFold and said no results are
reported. The item stays in the frozen set with D-031 attached; results for this stratum are
reported per item by name (`ut-004` clean; `ut-003` mis-specified), never as 2 of 2.

**The cause is corpus composition, not a gap in the eval.** 13.4M characters of related-work
sections name nearly every term in machine learning at least once, so a 150-paper corpus of
research papers has almost no genuinely absent topics to ask about. Any corpus of this kind
would behave the same way. The finding is about what this corpus can support, and it is
worth more stated than worked around.

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

## Why the set is 69, not 100 — and what that is *not* evidence of

A reader seeing 69 of 100 will assume the corpus could not support the targets. **It could.**
Almost the whole shortfall is defective tooling, not a scarce corpus:

| Stratum | Target | Final | Cause | Kind |
|---|---:|---:|---|---|
| `single_paper_factual` | 55 | **43** (29 strong / 14 weak gold) | gold chunks selected on the wrong property; 3 had reference lists as gold | **construction defect** |
| `multi_hop` | 20 | **3** | gold on the wrong property, one hallucinated answer, three false presuppositions, one degenerate answer, one dropped qualifier, five with a paper contributing nothing | **construction defect** |
| `unanswerable_topic` | 4 | **2** | only two topics survive paraphrase-aware screening of 5,401 chunks | **genuine corpus limit** |
| `unanswerable_attribute` | 11 | **11** | — | — |
| `ambiguous` | 10 | **10** | — | — |

**Only the 4→2 in `unanswerable_topic` is the corpus's fault.** 13.4M characters of
related-work sections name nearly every term in machine learning, so a research corpus has
almost no genuinely absent topics to ask about; any corpus of this kind behaves the same way.

Every other missing item was available and was lost to a check or a selector that measured
the wrong thing — the selector ranked gold chunks by figure density while the check validated
on figures, so both optimised toward the same wrong property and settled on bibliographies
(DECISIONS D-023, ninth instance). The corpus offered 314 bridgeable pairs and 142 factual-
eligible papers. The tooling could not tell a good gold chunk from a reference list.

This framing is less flattering to the tooling and more accurate, which is the trade it is
recorded for. Restating it the other way — "the corpus supports only 73 items" — would be a
false claim about the data, and would have removed any reason to fix the selector.

---

## The tightening rule

**A stricter check is an improvement only if what it newly rejects is actually wrong.**

Every check here has been tightened at least once, and tightening has produced false
rejections as often as it has caught real defects. Two instances, both of which looked like
progress at the moment they were made:

* The lexical-leakage metric was tightened from nothing to unigram bag overlap, and culled
  **36 of 55** factual questions. Reading them showed the overlap was dominated by
  unavoidable proper nouns — *"What is the expense per 1,000 evaluations for Gemini 3.1
  Flash-Lite?"* scored 0.80 while being correctly paraphrased.
* The premise check was tightened from substring to word-boundary matching, correctly
  catching `cifar-10` inside `cifar-100` — and immediately rejecting an item asking *"how
  much RAM was used"* of a paper reporting *"256GB system memory"*, which is the paraphrase
  the construction rules demand.

So the rule is procedural, not a sentiment: **after tightening a check, read what it newly
rejects before trusting the new number.** A cull rate that rises is not evidence the check
improved; it is evidence the check changed. The two are distinguishable only by inspection,
and in both cases above the pooled rate looked like a finding and was an artifact.

The corollary is that a check's exclusions need a *criterion*, not a list. `RAM`, `GPU`,
`FLOPS` and `AUC` were each added to the premise check's stop list after causing a false
positive, which encodes the terms that happened to be hit and leaves `TPU`, `MACs`, `IoU`
and `BLEU` untested. `make audit-entities` prints every entity the check asserts across the
whole set, so the next borderline term surfaces as a line in a list rather than as a
rejection weeks later.

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
`ConstructionReport.diagnosis()` names which failure mode dominates.

### The lexical-leakage check measured the wrong thing

The first full run culled **39 of 55** factual questions. The construction rule forbids
*"verbatim reuse of distinctive multi-word phrases"*; the implementation measured unigram
bag overlap, which is a different thing and wrong in a specific direction — it penalises
questions for naming entities that have no paraphrase:

| overlap | question | verdict |
|---:|---|---|
| 0.80 | "What is the expense per 1,000 evaluations for Gemini 3.1 Flash-Lite?" | correctly paraphrased; "expense" for "cost" |
| 0.60 | "What condition must hold for the tail value r under round-to-nearest rounding?" | technical term, no synonym exists |
| 0.57 | "…top-1 performance of S-Adam on the ImageNet dataset" | you cannot ask about ImageNet without saying ImageNet |

Replaced with contiguous n-gram overlap plus longest verbatim run. Re-measured against real
380-token chunks: **0 of 12 culled, median phrase overlap 0.00, longest verbatim runs of one
to four words.** The drafter was never lifting phrasing. Two-thirds of the stratum would
have been discarded to fix a problem that did not exist.

The correct instinct here was *not* to relax the threshold. A threshold that culls too much
and a metric that measures the wrong quantity look identical from the cull rate alone — the
only way to tell them apart was to read the culled questions.

### Do the checks fire? Both of them, on real output

Same question as everywhere else in the D-023 family. Answered rather than assumed:

* **`banned_phrasing`** — the original prompt (the one that produced eight research
  proposals) replayed through the detector fired on **2 of 6** real generations in the first
  probe and **1 of 6** in a second, independent probe after the reporting fix
  (`integrate` + `how can`). Not inert — and worth probing precisely because the
  construction report had once shown `banned_phrasing: 39` for culls that were nothing of
  the kind, which is a number that stops anyone from asking whether the check works
  (DECISIONS D-023, sixth instance).
* **phrase overlap** — a verbatim lift from a real chunk is caught; a necessary entity name
  is not; three real drafted questions pass.

The 2-of-6 is worth reading carefully: the regex catches the *lexical* tell, and the other
four old-prompt failures were semantic — comparative in wording, unanswerable in substance.
Those are what the necessity check exists for. **Neither check subsumes the other**, and a
drafter policed by only one of them would leak in whichever direction it does not look.

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
make verify-evals FILE=evals/datasets/phase4.json     # verify interactively
python -m evals.verify_cli evals/datasets/phase4.json --report    # progress only
python -m evals.verify_cli evals/datasets/phase4.json --stratum multi_hop
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

---

## Hazard 5 — the pooled clause

A multi-hop answer is two halves joined by a contrast: *"Paper A reports X, whereas Paper B
reports Y."* Every check in the chain originally pooled their claims into one set, and pooling
cannot distinguish **this paper supports its own claim** from **this paper's chunks contain the
other paper's numbers**.

Three defects lived in that gap, all three found by a human reading gold against answers, none
findable by the checks as written:

* **The degenerate clause.** *"…whereas Paper 30120 reports a hidden dimension h of the sparse
  autoencoder"* — a variable name where the question asked for a value. The item reads as
  answerable and grades every value correct. Its gold then passed because 30120's chunks
  contain 1024 and 768, which are the *first* clause's figures.
* **The false contrast.** *"What distinct noise power values and slot durations do the two
  papers report"* asserts the values differ; both papers report 0.1 s. Grounding passes,
  because each half is individually true. The falsehood is in the question's presupposition.
* **The conjunctive discriminator.** *"Which of the two evaluates on public network packet
  traces **while** using a four-phase transmission protocol"* — one paper has the traces, the
  other the protocol, neither has both. `false_premises` needs an entity absent from *all*
  papers, `universal_premises` needs one present in all, and `premise_severity` passes
  discriminative forms unconditionally. The missing case is *no single paper has all of them*.

Attribution is now scoped to the clause naming each paper. The conjunction is the one that
needs a model call: its conditions are descriptive phrases, not named entities, so no string
check can verify them against a paper. `evals/multihop.py:some_paper_satisfies` asks each paper
whether it satisfies every part of the condition, and a failure to reach the model leaves the
item inadmissible rather than admitted.

**What the clause split cannot see.** It keys on paper ids in the answer. An answer that names
its papers descriptively — *"the hematological malignancy cytology study … whereas the
pancreatic cancer screening study"* — has no clause boundary a string can find, so the
clause-scoped checks decline rather than reject. Declining is deliberate: an earlier version
read the absence of an id as "the answer never mentions this paper" and rejected a sound item.
The gap is real and unautomated; those items rest on human verification.

---

## Gold sets are minimal, and why that is a metric decision

`gold_chunks_for` returns the three best chunks per paper, so a two-paper item carried up to
six. The surplus is *interchangeable*, not dead — three chunks of one paper each supporting the
same four benchmark names. `unsupported_chunks` cannot see it, because every one of them
supports something.

It matters because **Recall@k divides by the size of the gold set.** Three interchangeable
chunks turn one correct retrieval into a fractional credit and dilute MRR@k in the retriever's
favour, so a gold set that is generous is not neutral — it flatters the system under test.
`make prune-gold` reduces each set by greedy cover over the answer's claims, keeping at least
one chunk for every paper whose clause contributes. Across the set it removed 32 chunks from 24
items; multi-hop gold went from six chunks to two or three.

Pruning **removes and never adds**. Where the corrected ineligibility rule (D-027) newly admits
a better chunk, the tool names the item and stops: a human verified specific gold, and swapping
it silently would invalidate that verification while the item still reads as accepted.
Re-selection is the reviewer's call.

---

## What confirmation needs, and what an excerpt cannot give

Human review reported three multi-hop items as unconfirmable — *"1,634 / 189 / 183,098 appear
in no visible excerpt"*, *"no chunk visibly contains Qwen 2.5 7B"*. All three claims were
present. `evals/verify_cli.py` shows a window, and each string sat outside it: 1,634 is 900
characters into its chunk, "Qwen2.5 7B" is the last sentence of a hyperparameter appendix,
"Llama 3.2 1B" is a table row.

Two lessons, one for the tool and one for the reader. `make gold-report` prints each gold chunk
in full and locates every claim string with its surrounding context, because **a reviewer
cannot confirm containment against a truncated view** — and an unconfirmable item gets rejected,
which is the expensive direction to be wrong in. And the check and the human were searching
different strings: containment normalises thousands separators, so it matches `183098` where a
reader greps `183,098` and finds nothing.

---

## The frozen set: 69 items, and what the multi-hop stratum can no longer support

**Frozen at `evals/datasets/phase4.json`, sha `de699d68`, on 2026-09-10.** 43 factual / 3
multi-hop / 2 unanswerable-topic / 11 unanswerable-attribute / 10 ambiguous. The three freeze
conditions hold: `make verify-dataset` reports zero disagreements on all 69; the defect matrix
passes on hand-authored, reviewed fixtures; every shortfall above is attributed. No further
rebuild cycles — the remaining defects in the tooling are recorded, not chased.

**Multi-hop is reported as a case study, never as a rate.** Three items yield 0/33/67/100%,
which is not a measurement. It is reported the same way `unanswerable_topic` (n=2, effectively
n=1 clean after D-031) is: per item, pass/fail, with n stated inline every time.

**State the consequence, so nobody infers the original claim from a table that no longer
supports it.** The spec's multi-hop stratum existed to measure an *orchestration delta* — the
part of v3's improvement attributable to LangGraph routing rather than to retrieval. At n=3:

* Recall@k and MRR@k over multi-hop gold are undefined as rates. Three items cannot bound
  either within anything narrower than the whole interval.
* An orchestration delta cannot be estimated. The stratum that would have carried it is a
  case study.
* **The v2.1-vs-v3 comparison therefore rests on the factual (n=43), refusal (n=13, two
  sub-strata never merged) and ambiguous (n=10) strata.** Any statement that v3's graph
  orchestration improves multi-hop retrieval is unsupported by this set and must not be made.

This is a corpus-and-tooling outcome, not a design choice, and it is stated here rather than
softened: the multi-hop stratum was drafted at 20, verified down to 7, re-verified down to 4,
and lost one more to a qualifier check written after the fact. Every one of those losses was a
construction defect the tooling missed and a human caught.

---

## Two multi-hop items rest on human judgement the tooling cannot replace

* The clause-scoped checks key on paper ids in the answer. `mh-003` names its papers
  descriptively, so those checks decline on it. Its containment is confirmed by
  `make gold-report`; its clause structure is confirmed by a reader.
* The qualifier check (`unaddressed_qualifiers`) sees hyphenated compounds. A qualifier
  written as a single word — "primary", "default", "largest" — is invisible to it.

---

## The gold diff was unanswerable, and positional ids are why

The regression report matched current gold against an earlier revision's gold on the source
paper pair, since item ids are positional and change on every rebuild. It matched every
survivor. It was still the wrong comparison: the revision it diffed against (`ee95155`) is not
the one the reviewer had verified, and the chunks the reviewer named as dropped
(`30232_0101`, `29628_0021`) appear nowhere in it — **three different gold sets exist across
three drafts**, and the diff could not identify which one a human had seen.

Recorded beside the id-instability finding because it is the same defect one level up: an
unstable id makes an *item* unaddressable across versions, and an unversioned verification
makes a *reviewed state* unaddressable across versions. What a reviewer accepted is a fact
about one artifact sha, and the sha was never attached to the review. Moot for the frozen set;
binding for any future one — verification records must carry the artifact sha they were made
against.

---

## Refusal accuracy and hallucinated-refusal rate are one number pair, never two numbers

The first human-scored 25 (`evals/runs/scores_human_de699d68.json`, rubric sha-stamped):

| stratum | n | outcome |
|---|---:|---|
| `unanswerable_attribute` | 8 | 8 `correct_refusal` |
| `unanswerable_topic` | 2 | 2 `correct_refusal` (case study, not a rate; `ut-003` is a mis-specified item — the agent was right, D-031 — so effectively n=1 clean) |
| `ambiguous` | 5 | 3 `correct_clarification`, 2 `silent_disambiguation` |
| `single_paper_factual` | 10 | 3 `correct_answer`, **7 `hallucinated_refusal`** |

**Refusal accuracy 10/10 (n=13 across both sub-strata) is not evidence of scope discipline
when it stands next to a hallucinated-refusal rate of 7/10 (n=10).** Read together, the two
numbers describe an agent that refuses readily — one that would score the same 10/10 on the
unanswerable strata by refusing everything. The unanswerable stratum on its own cannot
distinguish *knowing what is absent* from *giving up easily*; only the answerable stratum's
refusal rate can, and it says the agent gave up on 7 of 10 questions the corpus answers.

So the pair is published as a pair, in the same sentence, every time. `refusal_accuracy =
1.00 (n=13)` printed alone is exactly the near-perfect-metric trap the integrity rules forbid:
a perfect number on a stratum the system's failure mode passes for free.

**Where the seven refusals came from** (`make bypass-probe --from-scores …`, buckets decided
by the rubric's own grounding criterion — was the gold fact in what the agent retrieved):

| bucket | n | items | meaning |
|---|---:|---|---|
| retrieval | 5 | sp-007, sp-049, sp-032, sp-047, sp-030 | the gold chunk was not retrieved at k=5 or k=10; three without even the gold paper. Same under-anchoring as D-029's sp-003/sp-021: the paraphrase rule left nothing for the retriever to key on |
| generation | **1** | sp-036 | the gold fact *was* retrieved — "For Llama 3.2 models, we use context length 8192", a neighbour of the gold chunk in the same paper — the agent **quoted it in its refusal** and still said the question was unanswerable, because "the smaller family of language versions" did not resolve to Llama 3.2 for it |
| guardrail | 1 | sp-044 | refused before retrieval (D-029) |

The generation case is one item and is reported as one item. It is also the only one of the
seven where the system had everything it needed and refused anyway, which is why it is named
rather than folded into a rate — and why the bucket is decided by *fact in retrieved
context*, not *gold chunk retrieved*: the stricter criterion would have filed sp-036 under
retrieval and hidden the one generation failure in the set.

**Consequence for the judge experiment.** The judges must discriminate `hallucinated_refusal`
from `correct_refusal` above all else — 17 of the 25 human labels are one of those two, and
the pair is the headline. Per-arm agreement is reported on that pair separately from the rest.

---

## What is pinned, and the one thing that is not

Every input to a Phase 4 number can be named by hash and regenerated:

| input | pin |
|---|---|
| corpus | `e1be96d1` (`make verify-corpus`) |
| eval set | `de699d68`, 69 items (`make verify-dataset`) |
| judge sample | seed `20260910`, sha-stamped (`make judge-sample`) |
| rubric | sha stamped on every score sheet; a changed rubric refuses to score |
| agent prompt version | `v2`, recorded per run |
| the run | every item's question, gold, agent answer, and full retrieved chunk text (`evals/runs/v3_de699d68.json`) |
| every judge answer | per item, per stage, per arm (`evals/runs/judge_<arm>_<stage>.json`) |

**The judge's weights revision is the single link that is not pinned**, because no provider
exposes one. The Hugging Face router returns no revision, commit or checkpoint field for
`openai/gpt-oss-120b` and routes across eleven providers with their own serving stacks; the
OpenAI account lists `gpt-5.6-luna` with no dated snapshot alias. What *is* recorded is the
served model id per response, asserted to match the request, so a substitution is loud.

This is not the v2.1 failure. v2.1's `MRR@5 = 0.990` was unreproducible **and unlabelled**:
the dataset was deleted, its papers did not overlap the shipped corpus, and nothing recorded
what had been measured. The numbers here are reproducible on every dimension except one, and
the artifact names that dimension. A reader can re-run everything and attribute any drift to
exactly one cause. D-030.

---

## Hazard 6 — the screen verifies a string; the question asks about a concept

Twice now hand-scoring has caught what eight rounds of automated checking passed, and both
cases are the same gap seen from different sides. **`sp-036`**: the generator retrieved the
fact, quoted it, and refused. **`ut-003`**: the item's absent term was *AlphaFold*, its
question asked about *protein structure prediction*, and the corpus has OmegaFold doing
exactly that — **the agent was right and the item was wrong.** Hazard 1 is the question
paraphrased *below* the verified term; this is the question generalised *above* it. The rule
for the next set: verify absence of what the question asks about, not of the seed term
(D-031). This is not an argument against the checks. It is the measured size of what they
cannot see, and the reason a human-scored slice is not optional.

---

## Judge validation — the control that reads every other arm

Before any further spend, the generator (`gemini-3.5-flash-lite`) judged its own 25 answers
under the same two-stage prompt (`make judge-agreement`, $0). It agreed with the human on
**23 of 25** — `hallucinated_refusal` 7 of 7, `correct_refusal` 8 of 10 — and with
`luna-low` on **25 of 25 with each other**, missing the same two items (`ua-008`, `ut-003`) in the same way.

So a judge agreeing with the human 23 times in 25 has not demonstrated independence: the
generator judging itself does exactly that. On this sample the items are mostly easy to
score once the rubric is applied, and the experiment cannot separate an independent judge
from a self-preferring one. What it *can* say: no self-preference is visible at Q1 — the
generator called all seven of its own hallucinated refusals refusals — and the two
disagreements are the rubric's hedge sentence, read identically by two unrelated models.
Counts per cell, never percentages; at n=7 one item is fourteen points.

---

## The judge experiment, closed (D-032)

**Shipping judge: `gpt-5.6-luna` at `low`, Batch API, two-stage prompt.** It did not win.
Three complete sheets — the generator judging itself, `luna-low`, `luna-medium` — agree with
each other on 25 of 25, so agreement is undetermined among them; cost and pinnability decide,
and `low` is the cheapest of three indistinguishable options.

**Amendment 4's question is answered as unanswerable on this sample.** It asked whether a
free open judge is competitive with a paid one. What the experiment measured is that 25 items
scorable by a fixed rubric do not distinguish any judge from any other, including one with
maximal incentive to favour the system under test. That is a finding about the instrument.

Two numbers: **`medium` spent ~1.6× the output tokens of `low` and moved zero labels**, and
**the whole experiment cost ≈$0.0057 against the $5 ceiling.**

**Before and after the rubric amendment** (`make judge-agreement`, counts per cell, the
amendment named on every line it touches):

| sheet | before (v1) | after (v2, `ua-008` and `ut-003` re-judged) |
|---|---|---|
| human | reference sheet; labels unchanged under the v2 amendment, which codifies the scorer's reading | — |
| `flash-lite-self` (control) | 23 of 25 | 25 of 25 — after rubric v2 amendment, rescored `ua-008`, `ut-003` |
| `luna-low` | 23 of 25 | 25 of 25 — after rubric v2 amendment, rescored `ua-008`, `ut-003` |
| `luna-medium` | 23 of 25 | 25 of 25 — after rubric v2 amendment, rescored `ua-008`, `ut-003` |

A post-amendment 25 of 25 never appears without the amendment in the same sentence. Without
it, that number is the near-perfect-metric trap; with it, it says exactly what happened: the
rubric had a gloss two unrelated models read one way and a human read the other, the gloss
became a criterion, and the disagreement closed. The sample cannot tell the judges apart; it
can tell a rubric gloss from a rubric criterion.

---

## The variance estimate — designed around measured instability, not assumed stability

Two instruments produce every Phase 4 outcome number, and both are nondeterministic:

* **The generator.** D-014: five runs of one prompt gave five distinct strings; `temperature`
  is ignored and thinking cannot be disabled.
* **The judge-side model.** `tests/test_necessity_fixtures.py`, rewritten to three repeats per
  case, measured the same Gemini instrument on a real multi-hop pair at **1 of 3, then 2 of
  3** draws landing where the case was placed, with three distinct verdict patterns across
  three draws. The single-paper and neither cases were stable at 3 of 3. That is the number
  the design below is built around; a fixture tuned until it stopped moving would have hidden it.

**Design.** N = 3 complete runs of v3 over all 69 items (`make run-set` with `--tag r2`,
`--tag r3`; ~15 min each at the pacing rate, billed on the key's paid project — D-046), each scored in full by the shipping judge
(`gpt-5.6-luna` at `low`, two-stage, Batch). Every metric — the refusal pair, answer accuracy,
grounding, clarification, Recall@5, MRR — is then reported as **three values and their
spread** (min, max, std across runs), per stratum, with n. The judge is held fixed across
runs so the spread is the generator's; the judge's own variance is bounded separately by the
necessity fixture's split and, on the 25-item sample, by three sheets agreeing with each other on 25 of 25.

**Cost, estimated before submission** (`make judge-estimate` with `--all`): 115 requests per
run at the upper bound (69 q1 + up to 46 q3), ~$0.064 per run at batch rates, so ≤ $0.19 for
three; the q3 count is bounded above by the answerable items and in practice much lower — the
first run answered 3 of 10 sampled factual items. Against the $0.64 allocation for "100-item
full runs ×6".

**What N = 3 can and cannot say — stated before running it, not after.** The judge-side
instrument measured 1 of 3 and 2 of 3 on `genuinely_multi_hop` with three distinct verdict
patterns in three draws: on that case it is close to a coin flip, and it is the same
instrument this estimate uses. So the estimate is reported as **three draws with the spread
stated and no confidence interval** — n=3 cannot support one, and printing one would be
arithmetic dressed as inference. Three draws resolve a spread to one item in 69 (≈1.4 points
on a 69-item rate, ≈2.3 on the 43-item factual stratum) and nothing finer. If the spread on
the real metrics is narrow, that is said plainly as a useful finding cheaply obtained. If it
is wide, the metrics cannot support fine comparisons between arms, and that constrains how
the comparison arms are reported — decided here, in advance, rather than discovered when an
arm difference turns out to sit inside the spread.

**Recall@k does not appear in the README until the v2.1 arm has run.** The 0.267 measured on
the first run is v2.1's frozen retriever (D-002) against paraphrased questions; a reader
would take it as v3's retrieval quality. It becomes interpretable only beside the v2.1 re-run
on the same set: if v2.1 lands near 0.267, the number is a property of the eval set's
paraphrasing — D-029's retrieval surface measured across the whole set — and belongs to neither system;
if v2.1 lands materially higher, something changed despite the freeze, and that is a serious
finding. The pairing is enforced the same way the refusal pair is: `tests/test_docs.py`
fails if a Recall@ figure appears in the README without a v2.1 figure in the same table row
or sentence.

**The embedding-arm decision rule, fixed before the spread is known.** The embedding
comparison (BGE-large, 1.3 GB, versus a smaller model) exists to answer Phase 5's container
question. It runs only if the three-run max−min spread on factual Recall@5 is **narrower than
0.05** — roughly two items of 43, and below any delta worth changing a model for. If the spread
is 0.05 or wider, the sample cannot resolve the difference the arm would measure; the arm is
not run, no second index is built, and Phase 5 ships the smaller model with the sentence *"the
quality difference is smaller than this sample can measure, so we shipped the smaller model
and said so."* Decided on the spread, in advance, not on the arm's result.

---

## The retrieval spread and the configuration arms — judge-free, so available now

`make metrics-compare`. Retrieval needs no judge, so these landed while judging is blocked
(the OpenAI account was deactivated mid-experiment; see CLAUDE.md §6). Local batch runs,
single-user, n=43 factual items, k=5 retrieved.

| run | config | Recall@5 | over items reaching retrieval | MRR | gold paper in top-5 | guardrail-blocked |
|---|---|---:|---:|---:|---:|---:|
| r1 | v3 | 0.267 | 0.303 (n=38) | 0.175 | 21/43 | 5 |
| r2 | v3 | 0.267 | 0.287 (n=40) | 0.175 | 22/43 | 3 |
| r3 | v3 | 0.267 | 0.295 (n=39) | 0.175 | 22/43 | 4 |
| dense_only | BM25 fallback off | 0.267 | 0.311 (n=37) | 0.175 | 21/43 | 6 |
| section_filter | section filter on, keyword rule | **0.070** | 0.075 (n=40) | 0.060 | 10/43 | 3 |

**The spread on Recall@5 across three runs of the shipping system is 0.000** (max−min; MRR
likewise 0.000). Three draws, no interval. Retrieval given a question is deterministic here;
the only run-to-run variation is *which* items the scope guardrail blocks — 5, 3, 4 of 43 on
identical input — and blocked items are ones whose gold retrieval misses anyway, so the total
does not move. Two consequences: the embedding-arm rule (spread < 0.05) says **run it**, and
the gate's tolerance on retrieval is one item, not a percentage.

**The guardrail is itself nondeterministic.** The same 43 questions were blocked 5, 3, 4 and 6
times across four runs of configurations that do not touch the input layer. D-014's
non-determinism reaches the scope classifier; its false-refusal count on a fixed set is a
draw, and D-029's "5 of 43" is one such draw.

**`dense_only` is identical to the shipping system on retrieval.** With top_k=5 and a
fallback threshold of 5, dense always returns five hits, so BM25 never fires in the shipping
configuration either: the fallback's contribution to Recall@5 is exactly zero because it is
never invoked. D-015's separability question is answered by the trigger condition, not by the
retriever. Whether the fallback would help *if it fired* is a different experiment.

**`section_filter` collapses retrieval: 0.267 → 0.070, gold paper in the top-5 for 10 of 43.**
AUDIT §4.15 found the feature dead in v2.1 with an unvalidated section classifier; measured,
turning it on removes the gold chunk for most factual items — the classifier's labels do not
match where the facts are, or the keyword rule picks the wrong section, and this run cannot
tell which. It stays off, now on evidence rather than on suspicion.

---

## v2.1 on the same set, same generator — and Recall@5 becomes interpretable

`make run-v21`: published `main` at `8d3e67f`, corpus sha asserted equal, `gemini-3.5-flash-lite`
overridden at runtime, clone asserted pristine, 69 items in 32 minutes (6-hour timebox unused),
66 completed and 3 `out_of_scope`. Local batch run, single-user, n=43 factual, k=5.

| | v3 (r1–r3, spread 0.000) | v2.1 (`8d3e67f`) |
|---|---:|---:|
| Recall@5, factual | 0.267 | **0.233** |
| MRR | 0.175 | 0.196 |
| items with gold in top-5 | both systems: 12 · v3 only: 1 · v2.1 only: 0 · **neither: 30** |
| refused before retrieval | 5 (r1) | 3 — **the same three items** (`sp-003`, `sp-014`, `sp-026`) |

**v2.1 lands within one item of v3.** That is the first of the two branches stated before the
run: the number is a property of the eval set's paraphrasing — D-029's retrieval surface
measured across the whole set — and belongs to neither system. The same frozen retriever, driven by two
different orchestrations, misses the gold chunk on the same 30 of 43 factual items. Nothing
changed despite D-002's freeze; the freeze held, and the retrieval quality it froze is what
these questions expose. v2.1's MRR is slightly higher (0.196 vs 0.175) because its ReAct loop
issues several searches per question and the union is longer (median 18 chunks retrieved, max
28) — a different quantity from v3's single k=5 pass, and reported as such rather than
compared as though equal.

**v2.1 refuses the same questions v3's guardrail refuses.** Its own scope check short-circuited
`sp-003`, `sp-014` and `sp-026` — three of the five v3 blocked — before any retrieval. The
false-refusal surface is the *question*, produced by the paraphrase rule, not v3's input layer
as such; D-029 measured it on v3 first because v3 ran first.

With the v2.1 figure beside it, Recall@5 may now appear in the README, and
`tests/test_docs.py` requires the pairing on every line it appears.

---

## The embedding arm — the spread said run it, and the difference is outside the spread

`run_set --arm embedding_small`: the same graph over `BGEsmall_cs512`, an index built from the
same 5,401 chunks with `BAAI/bge-small-en` (384-dim, ~130 MB) instead of the frozen
`bge-large-en` (1024-dim, ~1.3 GB). Retrieved lists differ from the large-model run on 65 of
69 items, so the arm did what it says. Local batch, single-user, n=43 factual, k=5.

| | bge-large (v3, three runs, spread 0.000) | bge-small |
|---|---:|---:|
| Recall@5 | 0.267 | **0.174** |
| MRR | 0.175 | 0.105 |
| gold in top-5 | both: 8 · large only: **5** · small only: 0 · neither: 30 |

**The difference is five items of 43, against a measured spread of zero.** The rule set before
the spread was known said the arm runs only if the spread is narrower than 0.05; it was, the
arm ran, and the result is above the resolution the rule required: the large model retrieves
the gold chunk on five items the small one misses, and the small one adds none. So Phase 5's
container question has an evidence-backed answer in the *other* direction from the cheap one —
**ship bge-large**; the size cost buys a measured retrieval difference. Stated with its size:
five items on a 43-item stratum, one draw of the small arm, and 30 items that neither model
reaches at all.

---

## The variance decomposes: retrieval 0.000, outcomes nonzero (the Phase 2 carry, answered)

The carry read *"measure variance of the metric, not the string."* Three complete runs of the
shipping system over the frozen 69, each judged in full by `gpt-5.6-luna` at `low`, answer it
with a **decomposition** rather than a single number.

| | spread over r1/r2/r3 |
|---|---|
| **Retrieval** — Recall@5, MRR, factual (n=43) | **0.000** |
| **Outcomes** — correct answers (n=46 answerable) | 15 / 16 / 16 → **1** |
| wrong answers | 5 / 4 / 6 → **2** |
| hallucinated refusals, answerable | 26 / 26 / 24 → **2** |
| ungrounded answers | 0 / 0 / 0 → **0** |
| refusal accuracy, attribute (n=11) | 11 / 11 / 11 → **0** |
| `unanswerable_topic`, correct refusals (n=2) | 2 / 1 / 2 → **1** |
| ambiguous clarified (n=10) | 4 / 3 / 3 → **1** |

**The variance lives entirely in unpinned generation and judging; retrieval contributes none of it.** (Rescoped 2026-09-24: generation here is unpinned. `seed=0, top_k=1` makes the generator's output byte-identical on a 5-draw probe — D-042 — so the generation share is a property of the sampling configuration, not of the model.)
Identical input, identical index, identical query embedding — the retrieved list is the same
every time, and every run-to-run difference downstream comes from what the generator wrote and
what the judge made of it. (The one exception is *which* items the scope guardrail blocks — 5,
3, 4 across the three runs — which is generation-side too: the classifier is an LLM call.)

**The reporting rule for the rest of the phase, set by this decomposition:**

* **Outcome metrics carry their spread inline.** "15 of 46 correct (spread 1 over three runs)".
* **Retrieval metrics do not need it** — the spread is 0.000 and stated once, here.
* A difference between arms **no larger than the spread has not been shown to be a difference**.
  Three draws, no confidence interval; n=3 cannot support one.

### The arms, each one draw, measured against that spread

`make metrics-spread`. `*` marks a difference from r1 larger than the spread.

| metric | r1 | `dense_only` | `section_filter` | v2.1 |
|---|---:|---:|---:|---:|
| correct answers (n=46) | 15 | 13 (−2) * | 8 (−7) * | **6 (−9) ** |
| wrong answers | 5 | 6 (+1) | 1 (−4) * | **11 (+6) ** |
| hallucinated refusals | 26 | 26 (+0) | 37 (+11) * | 29 (+3) * |
| ungrounded answers | 0 | 1 (+1) * | 0 | 0 |
| refusal accuracy, attribute | 11 | 11 | 11 | 11 |
| ambiguous clarified (n=10) | 4 | 3 (−1) | 0 (−4) * | 0 (−4) * |
| ambiguous silently disambiguated | 5 | 6 (+1) | 1 (−4) * | 9 (+4) * |

`dense_only` moves nothing outside the spread except two items' worth of answer quality, which
is consistent with the retrieval finding that BM25 never fires at all. `section_filter` is worse
on every axis that matters. And v2.1 is the row to read next.

---

## v2.1 versus v3 — no retrieval gain, a measured outcome gain, and v3's own worst result

The headline comparison. Same frozen 69 items, same corpus (`e1be96d1`), same generator
(`gemini-3.5-flash-lite`), v2.1 a clone of published `main` pinned at `8d3e67f` with a clean
tree asserted before the run. **The retriever is frozen identical between them (D-002)** — the
same FAISS index, the same BGE model, the same query prefix. Local batch runs, single-user.
Retrieval `make metrics-compare`; outcomes `make metrics-spread`, judged by
`gpt-5.6-luna` at `low`.

### Retrieval: v3 shows no gain, and on two of three measures v2.1 is ahead

| n=43 factual | v3 (r1–r3, spread 0.000) | v2.1 |
|---|---:|---:|
| Recall@5 (chunk level) | **0.267** | 0.233 |
| MRR | 0.175 | **0.196** |
| gold paper in the top 5 | 21–22 of 43 | **28 of 43** |
| items where neither retrieves the gold chunk | **30 of 43, the same 30** | |

**v3 is not better at retrieval.** It edges chunk-level Recall@5 by one and a half items;
v2.1 leads on MRR and finds the gold *paper* in seven more cases. On a retriever frozen
identical between the two systems, that is what "within noise of each other" looks like, and
the pre-stated branch-1 reading holds: **the number belongs to the eval set's paraphrasing**,
not to either system. Thirty of 43 items are missed by both, which is the same statement from
the other side.

The MRR and gold-paper edge has a mechanism, and it is not a virtue: v2.1's ReAct loop issues
several searches per question and unions the results — median **18 chunks from 6 distinct
papers** per item, against v3's **5 chunks from 3**. A wider net contains the right paper more
often. It also contains more wrong ones.

### Outcomes: v3 is ahead, outside the spread, and the mechanism is the wider net

| n=46 answerable | v3 (r1 / r2 / r3) | v2.1 |
|---|---:|---:|
| correct answers | 15 / 16 / 16 (spread 1) | **6** |
| wrong answers | 5 / 4 / 6 (spread 2) | **11** |
| hallucinated refusals | 26 / 26 / 24 (spread 2) | 29 |
| ambiguous clarified (n=10) | 4 / 3 / 3 | **0** — 9 of 10 silently disambiguated |

Both answer-quality differences are larger than the three-run spread. **Of v2.1's 11 wrong
answers, 7 cite only papers that are not the anchor** — it retrieves widely, then answers
confidently from the wrong paper (`sp-022`: gold paper `2605.29713`, answered from
`2605.30189`; `sp-043`: gold `2605.29908`, answered from `2605.30213`). v3, given a narrower
context, refuses instead. On ambiguous items the contrast is total: v3 asks which paper is
meant 3–4 times in 10, v2.1 never does and picks one silently 9 times in 10.

So the honest summary, and it is two claims not one: **v3's retrieval is not better — the
measurement shows no retrieval gain — and v3's answers are better, by a margin outside the
spread, because the orchestration constrains what the generator may answer from.** That is
what v3 was rebuilt for: LangGraph orchestration, checkpointing, guardrails and observability.
It is *not* a retrieval improvement and must never be presented as one.

### Where v3 is worse — the brief asked, and this is the answer

**v3 refuses 26 of 46 answerable items (r1; 24–26 across three runs).** That is the single
largest failure in this artifact, it is worse than any number in v2.1's deleted benchmark
claimed to be, and D-029 explains it: the paraphrase rule that makes the eval honest strips the
vocabulary the scope classifier, the retriever and the generator all key on. v2.1 refuses 29 —
also badly, on the same questions — so the failure is shared and the eval set is implicated in
it. Neither system should be described as good at this set.

---

## The regression gate, and the demonstration that it fires

`evals/baseline_metrics.json`, written from the three complete runs: **value = their mean,
tolerance = their max−min spread.** Not a round number — the tolerance *is* the measurement of
how much this system moves when nothing changes.

| gated metric | baseline | tolerance | from |
|---|---:|---:|---|
| factual Recall@5 | 0.267 | **±0.000** | 0.267 / 0.267 / 0.267 |
| factual MRR | 0.175 | **±0.000** | 0.175 / 0.175 / 0.175 |
| correct answers (n=46) | 15.67 | ±1 | 15 / 16 / 16 |
| hallucinated refusals | 25.33 | ±2 | 26 / 26 / 24 |
| attribute correct refusals (n=11) | 11.0 | **±0.000** | 11 / 11 / 11 |

Direction is per metric: fewer hallucinated refusals is not a regression. A missing baseline
exits 2 — the gate refuses to run on a tolerance nobody derived, rather than passing.

**It has fired, on three separate things.** A gate that has only ever passed has not been
shown to gate anything, so all three are asserted rather than described:

1. **A real regression.** `make gate RUN=…_section_filter.json SHEET=…_section_filter.json`
   fails, naming four metrics: Recall@5 0.070 vs 0.267, MRR 0.060 vs 0.175, hallucinated
   refusals 37 vs 25.33, correct answers 8 vs 15.67. The `section_filter` arm is a regression
   the gate would have caught had someone shipped it.
2. **An injected regression.** `tests/test_gate.py::TestTheGateFiresEndToEnd` copies the
   baseline run, drops every retrieved chunk, and asserts the gate CLI exits 1. The same two
   steps run in `.github/workflows/ci.yml` — the baseline must pass, and the regressed copy
   must fail; if the injected regression ever passes, CI errors with "the gate PASSED an
   injected regression — it does not gate anything".
3. **The clean run.** The committed r1 artifact passes its own gate, so the gate is not simply
   failing everything.
