"""Retrieval quality: suppress bibliography, spread across sources.

WHY THIS MODULE EXISTS
----------------------
Plain top-k cosine search over this corpus has two failure modes that both
corrupt a reasoning trace, because a trace can only be as honest as the
evidence it records.

FAILURE 1 -- reference lists outrank clinical text.
    The single highest-scoring chunk for "recurrence risk for B2 thymoma
    after complete resection" was::

        "s for thymic malignancies. J Thorac Oncol 2011; 6(Suppl 3): S1743-8.
         49. Sugie C, Shibamoto Y, Ikeya-Hashizume C et al. Invasive thymoma..."

    That is a bibliography page. It wins because reference lists are
    keyword-dense -- packed with "thymoma", "postoperative", "invasive" --
    while containing no claims at all. 68 of 337 chunks (20% of the corpus)
    look like this. Citing one is worse than retrieving nothing: it looks
    like evidence and supports no assertion.

FAILURE 2 -- one document monopolises the results.
    For the same query, ranks 1-19 were ALL from the ESMO guidelines; the
    first non-ESMO chunk appeared at rank 20. Yet the scores ran 0.787 down
    to 0.739 -- a 0.048 spread. Those chunks are statistically tied, so
    "best 5" is close to arbitrary among them, and the tie-break happens to
    favour whichever document is most stylistically similar to the query.

    The clinical cost is specific: asked about B2 histology, the model never
    saw the WHO classification papers that *define* B2. It inferred the
    histotype's behaviour from ESMO's radiotherapy advice instead of reading
    the diagnostic criteria. The answer was defensible; the evidence base
    was lopsided.

THE FIX -- two independent, auditable passes:

  1. ``reference_list_score`` demotes (never silently deletes) chunks that
     look like bibliography.
  2. ``diversify`` enforces a per-document cap so no single source can take
     every slot while near-tied alternatives exist.

Both are deterministic and explainable. Every suppressed or demoted chunk is
reported, so a reviewer can see what retrieval chose NOT to show the model --
which is itself part of an auditable record.

PORTABILITY CONTRACT (ARIAPlatform)
-----------------------------------
Nothing here knows about thymoma. ``diversify`` operates on
``(id, document, metadata, similarity)`` tuples and a metadata key to group
by. When this graduates to ``aria-client`` it becomes
``aria_client/reasoning/retrieval.py``; the bibliography heuristic becomes
one pluggable "chunk quality" filter among several.
"""

import re
from typing import Dict, List, Optional, Tuple

# Numbered bibliography entries: "12. Smith J" / "7. Marx A"
_REF_ENTRY = re.compile(r"(?:^|\n)\s*\d{1,3}\.\s+[A-Z][a-zA-Z'\-]+ [A-Z]")

# Journal volume/year citation style: "2011;6" or "2014; 64"
_JOURNAL_CITE = re.compile(r"\b(?:19|20)\d{2};\s*\d+")

# Common venue names and identifiers in this corpus.
_VENUE = re.compile(
    r"\b(?:doi|PMID|J Thorac Oncol|Ann Thorac Surg|J Clin Oncol|"
    r"Histopathology|Lancet Oncol|Eur J Cardiothorac Surg|Cancer Res)\b",
    re.IGNORECASE,
)

# "et al." density is a strong bibliography signal.
_ET_AL = re.compile(r"\bet al\b", re.IGNORECASE)


def reference_list_score(text):
    """Heuristic score for "this chunk is a bibliography, not prose".

    Counts four independent signals. Reference lists trip several at once;
    ordinary clinical prose that merely cites a study trips at most one or
    two, so the signals separate cleanly in practice.

    Args:
        text: The chunk text.

    Returns:
        Non-negative int. Empirically on this corpus: prose scores 0-3,
        bibliography scores 6+. See ``REFERENCE_SCORE_THRESHOLD``.
    """
    return (
        len(_REF_ENTRY.findall(text))
        + len(_JOURNAL_CITE.findall(text))
        + len(_VENUE.findall(text))
        + len(_ET_AL.findall(text)) // 2  # halved: prose uses "et al." too
    )


#: At or above this score a chunk is treated as bibliography.
REFERENCE_SCORE_THRESHOLD = 6

#: How much to penalise a suspected reference list, as a **fraction of the
#: observed score range** rather than an absolute number.
#:
#: This used to be a flat 0.25, which silently assumed cosine similarity on
#: a 0-1 scale. Once cross-encoder reranking started feeding in raw logits
#: spanning ~16 points, a 0.25 demotion became a rounding error and the
#: bibliography filter stopped working -- the exact failure the rerank
#: module's docstring warns about. Expressing the penalty relative to the
#: candidate spread makes it scale-free, so it behaves identically whether
#: scores are similarities or logits.
REFERENCE_PENALTY_FRACTION = 0.25

#: Floor for the penalty when every candidate scores nearly the same, so a
#: degenerate range cannot collapse the demotion to zero.
MIN_REFERENCE_PENALTY = 1e-6


def diversify(
    candidates,
    n_results,
    group_key="source",
    max_per_group=2,
    penalty_threshold=REFERENCE_SCORE_THRESHOLD,
    penalty=None,
):
    """Re-rank candidates to suppress bibliography and spread across sources.

    Two passes, in order:

    1. **Demote bibliography.** Any chunk scoring at or above
       ``penalty_threshold`` has ``penalty`` subtracted from its similarity.
    2. **Cap per document.** Walk the demoted ranking best-first, taking a
       chunk only while its group is under ``max_per_group``. If that does
       not fill ``n_results``, relax the cap and backfill — a thin corpus
       should degrade to plain top-k, not return short.

    Args:
        candidates: ``[(chunk_id, document, metadata, similarity), ...]``,
            typically the top ~40 from the vector store. Order irrelevant.
        n_results: How many to return.
        group_key: Metadata field defining a "document".
        max_per_group: Max chunks from one document before backfill.
        penalty_threshold: Bibliography score at which to demote.
        penalty: Absolute amount subtracted from demoted chunks. When
            ``None`` (the default) it is derived from the candidate score
            range via ``REFERENCE_PENALTY_FRACTION``, which keeps the
            demotion meaningful for both cosine similarities and
            cross-encoder logits.

    Returns:
        ``(selected, report)`` where ``selected`` is a list of
        ``(chunk_id, document, metadata, adjusted_similarity)`` sorted best
        first, and ``report`` is a dict describing what retrieval did:
        ``demoted`` (count), ``backfilled`` (count), ``groups``
        (group -> number selected). The report exists so the decision is
        auditable rather than silent.
    """
    if penalty is None:
        sims = [sim for _, _, _, sim in candidates]
        span = (max(sims) - min(sims)) if sims else 0.0
        penalty = max(span * REFERENCE_PENALTY_FRACTION, MIN_REFERENCE_PENALTY)

    scored = []
    demoted = 0
    for chunk_id, doc, meta, sim in candidates:
        ref_score = reference_list_score(doc or "")
        is_ref = ref_score >= penalty_threshold
        if is_ref:
            demoted += 1
        scored.append(
            {
                "chunk_id": chunk_id,
                "doc": doc,
                "meta": meta,
                "sim": sim,
                "adj": sim - (penalty if is_ref else 0.0),
                "is_ref": is_ref,
            }
        )

    scored.sort(key=lambda c: c["adj"], reverse=True)

    # Pass 2: take best-first under the per-document cap.
    selected: List[dict] = []
    counts: Dict[str, int] = {}
    for cand in scored:
        if len(selected) >= n_results:
            break
        group = str(cand["meta"].get(group_key, "?"))
        if counts.get(group, 0) >= max_per_group:
            continue
        counts[group] = counts.get(group, 0) + 1
        selected.append(cand)

    # Backfill: the cap must never cause a short result set.
    backfilled = 0
    if len(selected) < n_results:
        chosen = {c["chunk_id"] for c in selected}
        for cand in scored:
            if len(selected) >= n_results:
                break
            if cand["chunk_id"] in chosen:
                continue
            selected.append(cand)
            backfilled += 1
            group = str(cand["meta"].get(group_key, "?"))
            counts[group] = counts.get(group, 0) + 1

    selected.sort(key=lambda c: c["adj"], reverse=True)

    report = {
        "demoted": demoted,
        "penalty": penalty,
        "backfilled": backfilled,
        "groups": counts,
        "considered": len(candidates),
    }
    return [
        (c["chunk_id"], c["doc"], c["meta"], c["adj"]) for c in selected
    ], report
