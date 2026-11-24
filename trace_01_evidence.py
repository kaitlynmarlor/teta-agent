"""Step 1: Evidence identity — fixing the ambiguous ``[Source N]`` citation.

No LLM, no network. This demonstrates the single idea everything else rests on:

    A citation must be a STABLE ADDRESS, not a POSITION IN A LIST.

Run:  python trace_01_evidence.py
"""

from teta_agent.evidence import CitedWithin, EvidenceLedger

# ── Fake retrieval results ─────────────────────────────────────
# Two searches that OVERLAP — the ESMO chunk comes back in both.
# This overlap is the whole point: watch what happens to its label.

SEARCH_1 = {
    "query": "postoperative radiotherapy indications stage II thymoma",
    "hits": [
        ("2015-ESMO-Guidelines.pdf::0042", "Postoperative radiotherapy may be considered "
         "in stage II disease with high-risk features...",
         {"source": "ESMO Clinical Practice Guidelines", "author": "Girard et al.",
          "year": "2015", "page": "8", "printed_page": "v47"}, 0.78),
        ("Marx_2015_WHO_4th_ed_Classification.pdf::0013", "Type B2 thymoma is composed of "
         "plump epithelial cells with vesicular nuclei...",
         {"source": "WHO 4th Edition Summary", "author": "Marx et al.",
          "year": "2015", "page": "4", "printed_page": "1386"}, 0.71),
    ],
}

SEARCH_2 = {
    "query": "WHO type B2 thymoma recurrence risk after complete resection",
    "hits": [
        # SAME chunk as search 1, but here it is the FIRST hit, not the first.
        ("2015-ESMO-Guidelines.pdf::0042", "Postoperative radiotherapy may be considered "
         "in stage II disease with high-risk features...",
         {"source": "ESMO Clinical Practice Guidelines", "author": "Girard et al.",
          "year": "2015", "page": "8", "printed_page": "v47"}, 0.83),
        ("Marx_2022_WHO_5th_ed_Review.pdf::0088", "Recurrence rates after complete resection "
         "vary by histotype; Loehrer et al. reported a 57% response rate...",
         {"source": "WHO 5th Edition Review", "author": "Marx et al.",
          "year": "2022", "page": "9", "printed_page": "208"}, 0.69),
    ],
}


def show_old_way():
    """The flawed positional scheme from rag_chat.py."""
    print("=" * 68)
    print("❌ THE OLD WAY — [Source N] is positional, per-search")
    print("=" * 68)

    for n, search in enumerate((SEARCH_1, SEARCH_2), start=1):
        print(f"\nSearch {n}: \"{search['query'][:50]}...\"")
        for i, (chunk_id, _text, meta, _score) in enumerate(search["hits"], start=1):
            print(f"   [Source {i}] {meta['source']}  ← chunk {chunk_id[-4:]}")

    print("\n⚠️  THE BUG: the ESMO chunk is [Source 1] in search 1,")
    print("    and ALSO [Source 1] in search 2 — but in search 2 that label")
    print("    now competes with a different document set. If the model")
    print("    writes 'per [Source 1]', WHICH search was it talking about?")
    print("    The label is meaningless outside the search that produced it.")
    print("\n⚠️  SECOND BUG: look at the WHO 5th Ed passage. It says")
    print("    'Loehrer et al. reported a 57% response rate'. If the model")
    print("    writes '[Source 2] reports 57%', that is WRONG — Source 2 is")
    print("    the WHO review; Loehrer is a study it cites. We do not hold")
    print("    the Loehrer paper at all.")


def show_new_way():
    """Stable conversation-scoped labels via the ledger."""
    print("\n" + "=" * 68)
    print("✅ THE NEW WAY — [E_n] is a stable address for the whole case")
    print("=" * 68)

    ledger = EvidenceLedger()

    for n, search in enumerate((SEARCH_1, SEARCH_2), start=1):
        print(f"\nSearch {n}: \"{search['query'][:50]}...\"")
        for chunk_id, text, meta, score in search["hits"]:
            before = len(ledger)
            ref = ledger.add(chunk_id, text, meta, similarity=score,
                             first_seen_in=f"search_{n}")
            status = "reused ♻️ " if len(ledger) == before else "new     "
            print(f"   [{ref.label}] {status} {ref.citation()}")

    print("\n✅ The ESMO chunk is [E1] in BOTH searches. One chunk, one label,")
    print("   for the entire conversation. Re-ranking cannot move it.")

    print("\n── Resolving labels to URIs ─────────────────────────────────")
    for ref in ledger.all():
        print(f"   {ref.label} → {ref.uri}")

    print("\n   These URIs are literally the `evidenceRefs` array in ARIA's")
    print("   reasoning-node-event schema. No translation needed later.")

    # ── The cited-within distinction ───────────────────────────
    print("\n" + "=" * 68)
    print("✅ SECOND-HAND CLAIMS — 'as cited in', kept separate")
    print("=" * 68)

    # Look the passage up by chunk id rather than hardcoding a label --
    # dedup means label numbers are NOT the same as hit counts.
    host = ledger.by_uri(
        "chroma://teta_guidelines/Marx_2022_WHO_5th_ed_Review.pdf::0088"
    )
    loehrer = CitedWithin(described_as="Loehrer et al.", within_label=host.label,
                          note="57% response rate")
    print(f"\n   We retrieved : [{host.label}] {host.citation()}")
    print(f"   It credits   : {loehrer.render()}")
    print("\n   A clinician now sees the 57% figure is SECOND-HAND —")
    print("   reported by the WHO review, not read from Loehrer directly.")
    print("   The old [Source N] notation collapsed these into one thing.")

    return ledger


def show_page_fix():
    """Printed page numbers vs PDF indices — detected from the real corpus."""
    print("\n" + "=" * 68)
    print("✅ PAGE NUMBERS — cite what the journal printed")
    print("=" * 68)
    print("""
The chunker recorded `page = i + 1`, the PHYSICAL position in the PDF.
But these are journal reprints — the printed page numbers are offset:

   document                        pdf p1  ->  printed
   ─────────────────────────────────────────────────────
   Marx 2014 ITMIG Consensus         1     ->   596
   Marx 2015 WHO 4th ed              1     ->   1383
   Marx 2022 WHO 5th ed              1     ->   200
   ESMO 2015 Guidelines              1     ->   v40     (supplement style)

Citing "page 3" of the ITMIG consensus is not imprecise — it is WRONG.
No such page exists in the published article, so no one can check it.

`teta_agent/pages.py` detects the offset from each page's running header,
takes the MEDIAN across the document (robust to stray numbers in titles
and reference lists), and stores both numbers on every chunk.""")

    print("── Citations now carry both ─────────────────────────────────")
    ledger = EvidenceLedger()
    for chunk_id, text, meta, score in SEARCH_1["hits"] + SEARCH_2["hits"]:
        ledger.add(chunk_id, text, meta, similarity=score)
    for ref in ledger.all():
        print(f"   [{ref.label}] {ref.citation()}")

    print("\n   Printed page = what a clinician cites and can look up.")
    print("   PDF page     = how we find it in the local file.")
    print("   Both are kept: human citation and machine address have")
    print("   different stability requirements.")


def main():
    show_old_way()
    ledger = show_new_way()
    show_page_fix()

    print("\n" + "=" * 68)
    print("WHY THIS MATTERS FOR THE REASONING TRACE")
    print("=" * 68)
    print("""
A reasoning trace filters steps by asking "did the final answer depend on
this step?" To answer that mechanically you need evidence you can POINT AT.

  • Positional [Source N] → cannot be pointed at. "Source 3" of which
    search? The link between answer and evidence is unrecoverable.

  • Addressed [E_n] → the model writes [E1], we look up E1's URI, and we
    check whether a tool call in the recorded trace actually produced that
    URI. Verifiable, mechanically, with no LLM in the loop.

So fixing the citation bug is not a side quest — it is the PRECONDITION
for a filterable trace.
""")
    print(f"Ledger holds {len(ledger)} unique passages "
          f"from {len(set(r.source for r in ledger.all()))} documents.")


if __name__ == "__main__":
    main()
