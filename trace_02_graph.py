"""Step 2: The trace as a dependency graph, and what filtering means.

No LLM, no network. Fixed inputs so the graph is small enough to hold in
your head.

The one idea: a trace is not a log. A log is "then this, then that". A trace
records WHICH STEPS FED WHICH CONCLUSIONS -- and that is what makes it
filterable.

Run:  python trace_02_graph.py
"""

from teta_agent.claims import populate_trace, render_clinician_report
from teta_agent.evidence import EvidenceLedger
from teta_agent.trace import EVIDENCE, ReasoningTrace

# ── Fixed inputs ───────────────────────────────────────────────
# Three searches. The third is deliberately useless -- watch it disappear.

SEARCHES = [
    (
        "t1",
        "postoperative radiotherapy stage II thymoma",
        [
            ("2015-ESMO-Guidelines.pdf::0042",
             "Postoperative radiotherapy may be considered in stage II...",
             {"source": "ESMO Clinical Practice Guidelines",
              "author": "Girard et al.", "year": "2015",
              "page": "9", "printed_page": "v48"}, 0.785),
            ("2015-ESMO-Guidelines.pdf::0038",
             "The global trend has been toward less frequent use of PORT...",
             {"source": "ESMO Clinical Practice Guidelines",
              "author": "Girard et al.", "year": "2015",
              "page": "9", "printed_page": "v48"}, 0.783),
        ],
    ),
    (
        "t2",
        "WHO type B2 thymoma prognostic grade",
        [
            ("Marx_2014_ITMIG_WHO_Consensus.pdf::0057",
             "It is an open question whether B2 thymomas are intermediate "
             "grade, or whether B1 and B2 share a prognostic grade...",
             {"source": "ITMIG Consensus Statement",
              "author": "Marx et al.", "year": "2014",
              "page": "13", "printed_page": "608"}, 0.738),
            ("Marx_2022_WHO_5th_ed_Review.pdf::0088",
             "Loehrer et al. reported a 57% response rate...",
             {"source": "WHO 5th Edition Review",
              "author": "Marx et al.", "year": "2022",
              "page": "9", "printed_page": "208"}, 0.712),
        ],
    ),
    (
        "t3",
        "myasthenia gravis perioperative management",
        [
            ("2015-ESMO-Guidelines.pdf::0091",
             "Perioperative management of myasthenia gravis requires...",
             {"source": "ESMO Clinical Practice Guidelines",
              "author": "Girard et al.", "year": "2015",
              "page": "16", "printed_page": "v55"}, 0.701),
        ],
    ),
]

# A plausible model answer. It cites P1, P2, P3 -- and never mentions P5
# (the myasthenia passage) or P4 except as a second-hand attribution.
ANSWER = """## Postoperative Radiotherapy Considerations

Postoperative radiotherapy may be considered in completely resected stage II
disease with aggressive histology such as type B2 [P1].

The global trend has been toward less frequent use of PORT in thymoma,
reserving it for high-risk cases [P2].

It remains an open question whether B2 thymomas should be classified as
intermediate grade, or whether B1 and B2 belong to the same prognostic
grade [P3].

Loehrer et al. reported a 57% response rate (as cited in [P4]).

Multidisciplinary tumour board review is advised before proceeding.
"""


def build():
    """Replay the searches into a ledger and a trace."""
    ledger = EvidenceLedger()
    trace = ReasoningTrace(run_id="demo-case-1")

    for turn, (tool_id, query, hits) in enumerate(SEARCHES, start=1):
        trace.record_search(tool_id, query, turn)
        for chunk_id, text, meta, score in hits:
            ref = ledger.add(chunk_id, text, meta, similarity=score,
                             first_seen_in=tool_id)
            trace.record_evidence(ref, tool_id, turn)

    return ledger, trace


def main():
    ledger, trace = build()

    # ── 1. The unfiltered graph ────────────────────────────────
    print("=" * 70)
    print("1. THE RAW TRACE — everything that happened")
    print("=" * 70)
    print(f"\n{len(trace)} nodes recorded:\n")
    for node in trace.all_nodes():
        parents = f" ← {node.parent_ids}" if node.parent_ids else ""
        print(f"   {node.node_id:5s} {node.kind:10s} "
              f"{node.summary[:44]:44s}{parents}")

    print("""
This is still just a list. What makes it a GRAPH is the arrow column:
each evidence node points back at the search that produced it. Those
arrows are the thing we are about to walk.""")

    # ── 2. Claims link prose to evidence ───────────────────────
    print("\n" + "=" * 70)
    print("2. PARSING THE ANSWER INTO CLAIMS")
    print("=" * 70)

    answer_id, claims, hallucinated, second_hand = populate_trace(
        trace, ANSWER, ledger, turn=3
    )

    print(f"\nFound {len(claims)} sentences that cite evidence:\n")
    for claim in claims:
        tag = "  (second-hand)" if claim.is_second_hand else ""
        print(f"   {claim.labels} {claim.text[:50]}...{tag}")

    if hallucinated:
        print(f"\n   ⚠️  cited but not in ledger: {hallucinated}")
    else:
        print("\n   ✅ every cited label exists in the ledger")

    print("""
Note the last sentence of the answer — "Multidisciplinary tumour board
review is advised" — produced NO claim node. It cites nothing, so there
is no edge to draw. It is advice, not a grounded assertion, and the
trace is honest about that.""")

    # ── 3. The filter ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("3. FILTERING — walk backwards from the answer")
    print("=" * 70)

    kept = trace.filtered_to(answer_id)
    dropped = trace.unreached(answer_id)

    print(f"\n   kept:    {len(kept)} nodes")
    print(f"   dropped: {len(dropped)} nodes\n")

    for node in dropped:
        print(f"   ✗ {node.node_id:5s} {node.kind:10s} {node.summary[:44]}")

    print("""
Search t3 and passage P5 are gone. Not because they were low quality —
P5 scored 0.701, a perfectly respectable match — but because NOTHING IN
THE ANSWER DEPENDS ON THEM. The myasthenia search was run, returned a
real passage, and the model did not use it.

That is the entire filter: reachability. No LLM judged relevance. Two
reviewers running this get byte-identical output.""")

    # ── 4. The clinician view ──────────────────────────────────
    print("\n" + "=" * 70)
    print("4. THE CLINICIAN VIEW")
    print("=" * 70)
    print()
    print(render_clinician_report(trace, answer_id, ledger,
                                  second_hand_ids=second_hand))

    # ── 5. ARIA payloads ───────────────────────────────────────
    print("\n" + "=" * 70)
    print("5. THE SAME GRAPH AS ARIA EVENTS")
    print("=" * 70)
    events = trace.to_aria_events(answer_id)
    print(f"\n{len(events)} reasoning.node payloads (filtered subgraph).")
    print("Two examples:\n")
    import json
    for event in events[:1] + events[-1:]:
        print(json.dumps(event, indent=2))
        print()

    print("""These validate against
external/ARIA/contracts/schemas/events/reasoning-node-event.schema.json
with no translation — `evidenceRefs` is already an array of the URIs the
ledger minted in Step 1. That is why evidence identity came first.""")


if __name__ == "__main__":
    main()
