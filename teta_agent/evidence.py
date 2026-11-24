"""Evidence identity: stable, addressable references to retrieved material.

WHY THIS MODULE EXISTS
----------------------
The original ``rag_chat.py`` prompt asked the model to cite ``[Source N]``.
That notation is ambiguous in a way that matters clinically:

    "[Source 3] reports a 57% response rate"

Does "Source 3" mean...
  (a) the third chunk *we retrieved* in *this particular search*, or
  (b) a study *cited inside* that chunk's text?

Both readings are natural, and the model mixes them. Worse, meaning (a) is
**positional** -- "3" only means something relative to one search's result
ordering. Re-run the search with a different ``n_results`` and "Source 3" now
points somewhere else. A citation that changes meaning between runs cannot be
audited.

THE FIX: two distinct, explicitly-named reference kinds.

  RETRIEVED PASSAGE  ->  [P1], [P2], ...   (passages we actually pulled)
  CITED-WITHIN       ->  (as cited in P1)  (a claim the passage attributes
                                            to some other study)

``P`` labels are allocated *per conversation*, not per search. P1 keeps
pointing at the same chunk for the lifetime of a case, across every search.
That is what makes it a durable address instead of a row number.

On the choice of the letter ``P``, and why ``E`` was rejected, see
``LABEL_PREFIX`` below -- it is a clinical-safety point, not a style one.

Each label resolves to a URI::

    chroma://teta_guidelines/2015-ESMO-Guidelines.pdf::0042

PORTABILITY CONTRACT (ARIAPlatform)
-----------------------------------
Nothing below knows about thymoma, ChromaDB queries, or clinical text. An
``EvidenceRef`` is "a stable label + a URI + display metadata". That is exactly
the shape ARIA's ``reasoning-node-event.schema.json`` expects in its
``evidenceRefs`` array (a list of URI strings).

When this graduates to ``aria-client``, this module becomes
``aria_client/reasoning/evidence.py`` with the ``chroma://`` scheme swapped for
whatever a given agent retrieves from (``s3://``, ``pubmed://``, ``file://``).
``EvidenceLedger.uris()`` is already the exact payload field.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from teta_agent.pages import format_page_citation

# Scheme for the TETA guideline corpus. In a general ARIAPlatform build this
# becomes a parameter, not a constant -- see the portability note above.
EVIDENCE_SCHEME = "chroma"
COLLECTION_NAME = "teta_guidelines"

# Prefix for citation labels: P1, P2, ... "P" for *passage*.
#
# WHY NOT "E" FOR EVIDENCE: in ESMO clinical practice guidelines, square
# brackets already mean "[level of evidence, grade of recommendation]" --
# e.g. "[II, E]" or "[IV, B]". Grade **E** is defined as "strong evidence
# against efficacy, never recommended". 61 chunks in this corpus use that
# notation, and the model copies it into answers verbatim, so "[E1]" sat in
# the same paragraph as "[II, E]". To an oncologist scanning quickly, a
# bracketed E primes for "contraindicated" -- the opposite of a neutral
# pointer.
#
# "P" is unused by ESMO grading (A-E) and by evidence levels (I-V), and it
# reads as what it addresses: a retrieved passage. Bare digits were also
# ruled out because they collide with Vancouver reference numbers ("[50]")
# that appear inside the source PDFs.
LABEL_PREFIX = "P"


def evidence_uri(chunk_id, collection=COLLECTION_NAME, scheme=EVIDENCE_SCHEME):
    """Build the canonical URI for one retrieved chunk.

    ``chunk_id`` is ChromaDB's id, which ``build_db.py`` formats as
    ``{filename}::{chunk_index:04d}`` -- already stable and content-addressed,
    so we can lean on it directly.
    """
    return f"{scheme}://{collection}/{chunk_id}"


@dataclass
class EvidenceRef:
    """One retrieved passage, with a stable label for the whole conversation.

    Attributes:
        label: Conversation-stable handle, e.g. ``"P1"``. This is what the
            model writes in its answer and what the renderer resolves.
        chunk_id: ChromaDB id, e.g. ``"2015-ESMO-Guidelines.pdf::0042"``.
        uri: Canonical address; goes straight into ``evidenceRefs``.
        source: Human-facing document name, e.g. ``"ESMO Clinical Practice
            Guidelines"``.
        author: e.g. ``"Girard et al."``.
        year: e.g. ``"2015"``.
        page: *Physical* PDF page the chunk starts on. Machine-resolvable:
            this is how you find the page in the local file.
        printed_page: *Printed* page label from the published document, e.g.
            ``"596"`` or ``"v41"``. This is what a clinician cites. Empty when
            the document has no detectable pagination. See ``pages.py`` for
            why these two differ and why we keep both.
        text: The passage itself. Kept in memory for prompting; deliberately
            NOT written into trace events (events carry the URI only, so the
            journal stays small and the corpus stays the single source of
            truth).
        similarity: Retrieval score. With reranking enabled this is the
            cross-encoder relevance score, not raw cosine similarity; see
            ``rerank.py``.
        first_seen_in: Label of the tool call that first surfaced this chunk.
        line_start: First line of the chunk on its page, 1-based, as
            extracted. A human-readable address -- NOT used to position
            highlights; see ``viewer.py`` for why.
        line_end: Last line of the chunk on its page.
        anchor_head: Opening text of the chunk, recorded at build time so
            the viewer can find the passage in the real PDF.
        anchor_tail: Closing text of the chunk, same purpose.
    """

    label: str
    chunk_id: str
    uri: str
    source: str = "Unknown"
    author: str = ""
    year: str = ""
    page: str = "?"
    printed_page: str = ""
    text: str = ""
    similarity: float = 0.0
    first_seen_in: Optional[str] = None
    line_start: int = 0
    line_end: int = 0
    anchor_head: str = ""
    anchor_tail: str = ""

    def citation(self):
        """Short human citation.

        e.g. ``ESMO Clinical Practice Guidelines (Girard et al., 2015) p.v41 (pdf p.2)``
        """
        who = ", ".join(part for part in (self.author, self.year) if part)
        who = f" ({who})" if who else ""
        return f"{self.source}{who} {format_page_citation(self.printed_page, self.page)}"

    def line_range(self):
        """Human-readable line span, or empty when unknown."""
        if not self.line_start:
            return ""
        if self.line_end and self.line_end != self.line_start:
            return f"lines {self.line_start}-{self.line_end}"
        return f"line {self.line_start}"

    def header(self):
        """The block header shown to the model when this passage is injected."""
        return f"[{self.label}] {self.citation()}"


class EvidenceLedger:
    """Allocates and remembers ``P`` labels for one conversation.

    The ledger is what makes labels stable: a chunk retrieved by three
    different searches gets ONE label, allocated the first time it is seen.
    Ask for the same ``chunk_id`` twice and you get the same ``EvidenceRef``
    back.
    """

    def __init__(self):
        self._by_chunk: Dict[str, EvidenceRef] = {}
        self._order: List[str] = []

    def add(self, chunk_id, document, metadata, similarity=0.0, first_seen_in=None):
        """Register a retrieved chunk, or return its existing ref if already known."""
        existing = self._by_chunk.get(chunk_id)
        if existing is not None:
            # Keep the best similarity we have seen, but never re-label.
            if similarity > existing.similarity:
                existing.similarity = similarity
            return existing

        ref = EvidenceRef(
            label=f"{LABEL_PREFIX}{len(self._order) + 1}",
            chunk_id=chunk_id,
            uri=evidence_uri(chunk_id),
            source=str(metadata.get("source", "Unknown")),
            author=str(metadata.get("author", "")),
            year=str(metadata.get("year", "")),
            page=str(metadata.get("page", "?")),
            printed_page=str(metadata.get("printed_page", "")),
            text=document or "",
            similarity=similarity,
            first_seen_in=first_seen_in,
            line_start=int(metadata.get("line_start", 0) or 0),
            line_end=int(metadata.get("line_end", 0) or 0),
            anchor_head=str(metadata.get("anchor_head", "")),
            anchor_tail=str(metadata.get("anchor_tail", "")),
        )
        self._by_chunk[chunk_id] = ref
        self._order.append(chunk_id)
        return ref

    def get(self, label):
        """Resolve a label like ``"P2"`` back to its ref, or ``None``."""
        for chunk_id in self._order:
            ref = self._by_chunk[chunk_id]
            if ref.label == label:
                return ref
        return None

    def by_uri(self, uri):
        """Resolve a URI back to its ref, or ``None``. Used by verification."""
        for chunk_id in self._order:
            ref = self._by_chunk[chunk_id]
            if ref.uri == uri:
                return ref
        return None

    def all(self):
        """Every ref, in allocation order."""
        return [self._by_chunk[chunk_id] for chunk_id in self._order]

    def uris(self):
        """All URIs -- the exact shape of ``evidenceRefs`` in a GMP event."""
        return [self._by_chunk[chunk_id].uri for chunk_id in self._order]

    def known_labels(self):
        """Set of allocated labels, for validating what the model cited."""
        return {self._by_chunk[chunk_id].label for chunk_id in self._order}

    def __len__(self):
        return len(self._order)


@dataclass
class CitedWithin:
    """A study the *passage* credits, which we did NOT retrieve ourselves.

    This is the other half of the ``[Source N]`` fix. When a chunk says
    "Loehrer et al. reported a 57% response rate", the model must be able to
    attribute that to Loehrer *without* implying we hold the Loehrer paper.

    We record it as a claim *mediated by* evidence we do hold::

        CitedWithin(described_as="Loehrer et al.", within_label="P1")

    Rendered as "(as cited in P1)". A clinician reading the trace can then see
    the claim is second-hand and judge it accordingly -- which is exactly the
    distinction the old ``[Source N]`` notation destroyed.
    """

    described_as: str
    within_label: str
    note: str = ""

    def render(self):
        """Human-readable attribution."""
        return f"{self.described_as} (as cited in {self.within_label})"


@dataclass
class RetrievalResult:
    """What one ``search_guidelines`` call produced."""

    query: str
    refs: List[EvidenceRef] = field(default_factory=list)
    latency_ms: int = 0
    status: str = "succeeded"

    def prompt_block(self, max_chars=1200):
        """Format retrieved passages for injection into the model prompt."""
        if not self.refs:
            return "(no passages matched this query)"
        parts = []
        for ref in self.refs:
            text = " ".join(ref.text.split())
            if len(text) > max_chars:
                text = text[:max_chars] + "..."
            parts.append(f"{ref.header()}\n{text}")
        return "\n\n---\n\n".join(parts)

    def uris(self):
        """URIs surfaced by this call -- becomes the tool call's evidence set."""
        return [ref.uri for ref in self.refs]
