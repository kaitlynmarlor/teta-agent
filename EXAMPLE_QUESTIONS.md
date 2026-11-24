# Example questions for `rag_chat.py`

Every question below was **probed against the actual vector store** before
being listed; the number in parentheses is the top cosine similarity
(`1 - distance`) over the 337 indexed chunks. Nothing here is invented.

Rough calibration for this corpus:

| Top similarity | Meaning |
|---|---|
| > 0.80 | strongly covered — expect well-grounded claims |
| 0.72 – 0.80 | covered, but thinner; watch for hedging |
| < 0.65 | **not in the corpus** — the retriever still returns its five best chunks |

That last row is the important one. A vector search *always* returns
results. Low similarity does not produce an error, it produces confident-
looking irrelevant passages. The trace is what exposes this.

---

## A coherent case (run these in order)

> 55-year-old with an anterior mediastinal mass and fatigable ptosis.
> Resected; pathology shows WHO type B2 thymoma, Masaoka-Koga stage II,
> close margins.

These are the questions a real MDT works through, in sequence. Running
them in order is what exercises the **cross-turn** behaviour — the ledger
keeps counting labels up (`P1`…`P5`, then `P6`…), and a passage pulled for
an early question keeps its label if a later answer cites it.

1. How is an anterior mediastinal mass worked up before resection? (0.761)
2. What are the histological criteria for WHO type B2 thymoma? (0.815)
3. What defines Masaoka-Koga stage II thymoma? (0.817)
4. What is the appropriate extent of surgery for thymoma? (0.729)
5. What are the indications for postoperative radiotherapy in stage II? (0.841)
6. How should myasthenia gravis be managed perioperatively? (0.753)
7. What follow-up imaging is recommended after resection? (0.805)
8. How is recurrent thymoma treated? (0.794)

**Measured chunk overlap** (same chunk retrieved by both questions — this
is where you will see label reuse rather than new labels):

- Q4 ∩ Q6 — 2 shared chunks
- Q1 ∩ Q7, Q6 ∩ Q7, Q7 ∩ Q8 — 1 shared chunk each

After Q6, type `summary`. After Q8, type `summary` again and compare — the
`EVIDENCE BASE` section shows passages accumulating citations across
questions rather than being re-listed.

---

## Questions that stress specific trace features

### Multi-source — claims resting on different documents
**"What distinguishes WHO type B2 from type B3 thymoma?"** (0.849, ITMIG +
WHO 4th)

The best single demonstration. Retrieval pulls from two documents that
frame the distinction differently. If the answer's claims rest on sources
that disagree, the trace makes that visible instead of burying it in
synthesised prose.

### Second-hand citations
**"What response rates have been reported for chemotherapy in advanced
thymoma?"** (0.718)

Response-rate figures in this corpus live *inside* review articles —
ESMO citing Loehrer, not the Loehrer trial itself. Expect the
`⚠️ SECOND-HAND` flag. This is the distinction the whole evidence layer
exists to preserve: "the review says 57%" and "57%" carry different
weight.

### Not in the corpus — the most important test
**"What is the role of immunotherapy in thymic carcinoma?"** (0.792)

Note the *high* similarity. The 2015 ESMO guidelines predate checkpoint
inhibitors in this setting, so there is no real answer here — but
retrieval happily returns five oncology-flavoured chunks that *look*
relevant. This is the failure mode a reasoning trace is for.

Watch for: claims with no citations, or an `⚠️ CITED BUT NOT RETRIEVED`
warning if the model invents a label. Then run `raw` — you will see
passages retrieved with nothing pointing at them.

**"How is thymoma managed in paediatric patients?"** (0.738) — same shape.

### Clearly off-topic — the floor
**"What is the treatment for type 2 diabetes?"** (0.578)

Worth running once to see the floor. Similarity collapses and all five
hits cluster within 0.004 of each other — statistically tied, so the
ranking among them is close to arbitrary. A well-behaved answer should
decline to answer; the trace should show no supported claims.

---

## What to look at after each answer

| Command | Use it to |
|---|---|
| `summary` | triage the whole case — exceptions first |
| `trace` | read one answer claim-by-claim with its evidence |
| `raw` | see the filter working: ✓ kept, ✗ retrieved-but-unused |
| `evidence` | every passage so far, with resolvable URIs |

The single most informative habit: after an answer, check the
`passages used` count in the status line. Retrieval returns 5; if only 2
are used, the other 3 were considered and declined — and `raw` shows you
exactly which.
