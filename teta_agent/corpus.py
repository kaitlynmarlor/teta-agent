"""What the guideline corpus covers -- and, more importantly, what it does not.

WHY THIS MODULE EXISTS
----------------------
Retrieval can only return what it holds, and a vector search never fails:
ask about something absent and you still get five topically-plausible
passages. Measured on this corpus (``EXAMPLE_QUESTIONS.md``):

    "What is the role of immunotherapy in thymic carcinoma?"   top sim 0.792
    "What are the histological criteria for WHO type B2?"      top sim 0.815

Those scores are close. The second question is well covered; the first has
**no answer here at all** -- the only treatment document is ESMO 2015, which
predates checkpoint inhibitors in this setting. Retrieval returned
oncology-flavoured prose and no signal that the real answer was missing.

That is why this module is not a confidence score. A score derived from
similarity would have called the immunotherapy question "high confidence",
because semantic similarity measures *topical* match and cannot detect that
the content is absent -- confidently wrong exactly when it matters.

What a model cannot infer from retrieved text is the shape of the library
the text came from. So we state it plainly, once, and pass it along.

THE ASYMMETRY THAT MATTERS CLINICALLY
-------------------------------------
    Histological classification : 3 documents, 2014 / 2015 / 2022
    Treatment                   : 1 document, 2015

Classification guidance is current. Every treatment statement this system
can make rests on a single guideline from 2015. Nothing in a retrieved
passage reveals that imbalance.

MAINTENANCE
-----------
Document titles, years, and counts are the single source of truth in
``CORPUS_DOCUMENTS`` and must mirror ``PAPER_INFO`` in ``build_db.py``.
Everything countable in the prose is derived from that mapping, so adding a
document cannot leave a stale number behind.

The editorial sentences -- which clinical domain a document answers, and
what happened in the field after it was published -- are hand-written,
because that is judgement rather than data.

**Add a document -> add it to ``CORPUS_DOCUMENTS`` and review
``KNOWN_GAPS``.** The failure mode here is staleness, which is visible and
fixable; the failure mode of a calibrated threshold is silent wrongness.

Passage counts are deliberately *not* baked in. The corpus is chunked into
``vectorstore_v2/`` (MedEmbed, 214 passages) with ``vectorstore_data/``
(bge, 337) kept as a fallback, so any hardcoded figure is wrong for one of
them and both change on a rebuild. Callers that know which store they
opened may pass ``n_passages`` to :func:`scope_text`.

PORTABILITY
-----------
Pure stdlib. Deliberately free of ChromaDB and pypdf so
``tools/adapters.py`` can import it without dragging build-time
dependencies into the orchestrator.
"""

#: Documents grouped by the clinical question they answer. Mirrors
#: ``PAPER_INFO`` in ``build_db.py``; grouped by domain because the
#: imbalance between those groups is the point of this module.
CORPUS_DOCUMENTS = {
    "classification": [
        ("ITMIG Consensus Statement", "2014"),
        ("WHO 4th Edition Summary", "2015"),
        ("WHO 5th Edition Review", "2022"),
    ],
    "treatment": [
        ("ESMO Clinical Practice Guidelines", "2015"),
    ],
}

#: Publication year of the most recent treatment-guidance document. Named
#: because it is the single most consequential limitation of the corpus.
TREATMENT_GUIDANCE_YEAR = max(
    year for _, year in CORPUS_DOCUMENTS["treatment"]
)

#: Topics the corpus provably cannot answer. Editorial, not derived: each
#: is a consequence of treatment guidance stopping at 2015.
KNOWN_GAPS = (
    "checkpoint inhibitors / immunotherapy in thymic epithelial tumours",
    "staging revisions published after {}".format(TREATMENT_GUIDANCE_YEAR),
    "paediatric-specific management",
)


def _render(documents):
    """Format one domain's documents as "Title YEAR, Title YEAR"."""
    return ", ".join(
        "{} {}".format(title, year) for title, year in documents
    )


def document_count():
    """Total documents in the corpus, across all domains."""
    return sum(len(group) for group in CORPUS_DOCUMENTS.values())


def year_range():
    """``(earliest, latest)`` publication year as strings."""
    years = [year for group in CORPUS_DOCUMENTS.values() for _, year in group]
    return min(years), max(years)


def scope_text(n_passages=None):
    """Prose handed to a synthesis model alongside retrieved passages.

    States facts rather than instructions, so the model reasons *from* the
    limitation instead of reciting it -- with one exception at the end,
    because "answer from the nearest passage anyway" is the specific
    failure this text exists to prevent.

    Kept short on purpose: this rides along on every call, so length is a
    real cost.

    Args:
        n_passages: Chunk count of the vector store actually in use, if the
            caller knows it. Omitted from the text when ``None`` rather than
            guessed, since the two stores on disk disagree.

    Returns:
        A single paragraph of plain text.
    """
    size = "{} documents".format(document_count())
    if n_passages is not None:
        size += ", {} passages".format(n_passages)

    return (
        "Corpus: {size}. "
        "Histological classification is covered by {classification}. "
        "Treatment is covered by {treatment} only. "
        "All treatment guidance therefore predates checkpoint inhibitors in "
        "thymic epithelial tumours, post-{year} staging revisions, and "
        "paediatric-specific recommendations. If a question concerns any of "
        "those, the corpus does not contain the answer -- say so rather than "
        "reasoning from the nearest available passage."
    ).format(
        size=size,
        classification=_render(CORPUS_DOCUMENTS["classification"]),
        treatment=_render(CORPUS_DOCUMENTS["treatment"]),
        year=TREATMENT_GUIDANCE_YEAR,
    )


#: Default prose form, with no passage count. Importable as a constant for
#: callers that have no store handle; use :func:`scope_text` when the
#: passage count is known.
CORPUS_SCOPE = scope_text()


def describe():
    """One-line human summary, for terminal output and the clinician view."""
    earliest, latest = year_range()
    return "{} documents, {}-{}; treatment guidance {} only".format(
        document_count(), earliest, latest, TREATMENT_GUIDANCE_YEAR
    )


def known_gaps():
    """Topics the corpus provably cannot answer, as a list."""
    return list(KNOWN_GAPS)
