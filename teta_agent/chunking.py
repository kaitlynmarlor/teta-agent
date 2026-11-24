"""Semantic chunking: split on meaning, not on character count.

WHY THIS MODULE EXISTS
----------------------
The original chunker cut every 1000 characters with 200 of overlap. That
boundary is blind to meaning, and on this corpus it produced two concrete
failures:

FAILURE 1 -- claims severed from their qualifiers.
    "Postoperative radiotherapy is recommended for stage III disease" and
    the sentence that follows, "...but not for completely resected stage I",
    can land in different chunks. Retrieve one without the other and the
    model reads an unconditional recommendation. In a clinical corpus the
    exception usually matters more than the rule.

FAILURE 2 -- diagnostic criteria split mid-list.
    WHO histotype definitions run as enumerations. A fixed cut takes B2's
    first three criteria and leaves the rest for the next chunk, so neither
    chunk actually defines B2.

THE FIX -- embed every sentence, cut where meaning shifts.
    Adjacent sentences about the same topic have similar embeddings. When
    cosine similarity between consecutive sentences DROPS sharply, the text
    has moved on to something else: that is a natural boundary. We cut at
    the sharpest drops rather than at arbitrary character offsets.

    The threshold is a *percentile* of the observed drops in each document,
    not a fixed number. Documents differ in how abruptly they change topic,
    and a constant cutoff that suits ESMO prose over-segments WHO tables.

LINE PROVENANCE
---------------
Every chunk records the line span it came from (``line_start`` /
``line_end``, 1-based, relative to its page). PDFs have no canonical line
numbers -- these are lines as extracted -- so they are an audit aid and a
display nicety, NOT the mechanism used to highlight text in the viewer.
The viewer re-finds the passage by searching the real PDF for its text,
because extraction-relative line numbers are not stable enough to drive a
highlight. See ``viewer.py``.

PORTABILITY CONTRACT (ARIAPlatform)
-----------------------------------
Nothing here knows about thymoma. ``semantic_chunks`` takes pages and an
``encode`` callable and returns dicts. When this graduates to
``aria-client`` it becomes ``aria_client/reasoning/chunking.py``.
"""

import re
from typing import Callable, Dict, List, Optional

import numpy as np

#: Sentences shorter than this are glued to their neighbour rather than
#: embedded alone. "Fig. 2." carries no topic signal, and letting it vote on
#: a boundary adds noise.
MIN_SENTENCE_CHARS = 25

#: Target chunk size in characters. Soft: a chunk may run over if no good
#: semantic boundary presents itself before MAX_CHUNK_CHARS.
TARGET_CHUNK_CHARS = 1000

#: Hard ceiling. MedEmbed truncates at 512 tokens (~2000 chars), so going
#: beyond this silently discards text from the embedding.
MAX_CHUNK_CHARS = 1800

#: Ceiling for a single pre-chunk unit. Must be strictly below
#: ``MAX_CHUNK_CHARS``: an indivisible unit larger than the chunk limit
#: would blow the ceiling on its own, and fragment-merging can add a
#: further ``MIN_SENTENCE_CHARS`` on top. The gap absorbs both.
MAX_UNIT_CHARS = MAX_CHUNK_CHARS - 300

#: Never emit a chunk smaller than this; merge it forward instead.
MIN_CHUNK_CHARS = 250

#: Cut at drops in the steepest N% of the document's similarity drops.
BREAKPOINT_PERCENTILE = 80

# Sentence splitting. Abbreviations common in this corpus that must NOT end
# a sentence: "et al.", "Fig. 2", "vs.", "e.g.", "i.e.", "approx.", and
# initials ("Marx A. Muller-Hermelink").
_ABBREV = (
    r"(?<!\bet al)(?<!\bFig)(?<!\bfig)(?<!\bvs)(?<!\be\.g)(?<!\bi\.e)"
    r"(?<!\bNo)(?<!\bDr)(?<!\bapprox)(?<!\bca)(?<!\bcf)(?<!\bref)"
    r"(?<!\b[A-Z])"
)
_SENT_END = re.compile(_ABBREV + r"(?<=[.!?])\s+(?=[A-Z(\[\u2022])")


class Sentence:
    """One sentence, with the line span it occupied on its page.

    Attributes:
        text: The sentence as extracted.
        page: Physical PDF page (1-based) the sentence starts on.
        line_start: 1-based line number within that page.
        line_end: 1-based line number where the sentence ends.
    """

    __slots__ = ("text", "page", "line_start", "line_end")

    def __init__(self, text, page, line_start, line_end):
        self.text = text
        self.page = page
        self.line_start = line_start
        self.line_end = line_end

    def __repr__(self):
        return (
            f"<Sentence p{self.page} L{self.line_start}-{self.line_end} "
            f"{self.text[:40]!r}>"
        )


def split_page_sentences(page_text, page_number):
    """Split one page into sentences, tracking which lines each came from.

    The line tracking is the fiddly part. We walk the page line by line,
    accumulating a character offset for each line start, then map every
    sentence's character span back onto the line it fell in. Doing it this
    way (rather than splitting line-by-line) means a sentence wrapped across
    four lines stays ONE sentence and correctly reports a four-line span.

    Args:
        page_text: Raw extracted text for one page.
        page_number: Physical PDF page number, 1-based.

    Returns:
        List of ``Sentence``.
    """
    lines = page_text.split("\n")

    # Character offset at which each line begins, and the line number there.
    offsets = []
    pos = 0
    for i, line in enumerate(lines, start=1):
        offsets.append((pos, i))
        pos += len(line) + 1  # +1 for the newline we split on

    def line_at(char_pos):
        """Which line number contains this character offset?"""
        found = 1
        for start, lineno in offsets:
            if start <= char_pos:
                found = lineno
            else:
                break
        return found

    # Split the page, keeping track of where each piece started.
    sentences = []
    cursor = 0
    for part in _SENT_END.split(page_text):
        if not part.strip():
            cursor += len(part)
            continue
        # Locate this part in the original text so offsets stay truthful
        # even though the splitter consumed the whitespace between parts.
        idx = page_text.find(part, cursor)
        if idx == -1:
            idx = cursor
        sentences.append(
            Sentence(
                text=part.strip(),
                page=page_number,
                line_start=line_at(idx),
                line_end=line_at(idx + len(part) - 1),
            )
        )
        cursor = idx + len(part)

    return _merge_fragments(_split_oversized(sentences))


def _split_oversized(sentences, limit=MAX_UNIT_CHARS):
    """Break up units too long to embed, cutting on line boundaries.

    Needed because tables have almost no sentence-ending punctuation. The
    Masaoka staging table on one ESMO page extracts as a single 3536-char
    "sentence" -- and MedEmbed truncates at 512 tokens (~2000 chars), so
    everything past the cut would be silently absent from the vector and
    therefore unsearchable. Splitting on newlines is crude but exactly right
    for tabular text, where each line is already a row.
    """
    out: List[Sentence] = []
    for sent in sentences:
        if len(sent.text) <= limit:
            out.append(sent)
            continue

        buffer: List[str] = []
        size = 0
        start_line = sent.line_start
        line_no = sent.line_start
        for raw_line in sent.text.split("\n"):
            if buffer and size + len(raw_line) > limit:
                out.append(Sentence(
                    text=" ".join(buffer),
                    page=sent.page,
                    line_start=start_line,
                    line_end=line_no,
                ))
                buffer, size = [], 0
                start_line = line_no
            buffer.append(raw_line.strip())
            size += len(raw_line) + 1
            line_no += 1
        if buffer:
            out.append(Sentence(
                text=" ".join(buffer),
                page=sent.page,
                line_start=start_line,
                line_end=min(line_no, sent.line_end),
            ))
    return out


def _merge_fragments(sentences):
    """Glue too-short sentences onto their neighbour.

    A bare "Fig. 2." or a stray header carries no topic signal. Embedding it
    alone produces a meaningless vector that can trigger a spurious
    boundary, so we absorb it into the following sentence (or the previous
    one, if it is last).
    """
    if not sentences:
        return []

    merged: List[Sentence] = []
    pending: Optional[Sentence] = None

    for sent in sentences:
        if pending is not None:
            sent = Sentence(
                text=pending.text + " " + sent.text,
                page=pending.page,
                line_start=pending.line_start,
                line_end=sent.line_end,
            )
            pending = None
        if len(sent.text) < MIN_SENTENCE_CHARS:
            pending = sent
            continue
        merged.append(sent)

    if pending is not None:
        if merged:
            tail = merged[-1]
            merged[-1] = Sentence(
                text=tail.text + " " + pending.text,
                page=tail.page,
                line_start=tail.line_start,
                line_end=pending.line_end,
            )
        else:
            merged.append(pending)

    return merged


# ── Boundary detection ─────────────────────────────────────────


def similarity_drops(vectors):
    """Cosine distance between each consecutive pair of sentences.

    A HIGH value means the text changed topic between those two sentences --
    a candidate cut point.

    Args:
        vectors: ``(n, dim)`` array of sentence embeddings.

    Returns:
        Array of length ``n - 1``.
    """
    if len(vectors) < 2:
        return np.array([])
    normed = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-10)
    sims = np.sum(normed[:-1] * normed[1:], axis=1)
    return 1.0 - sims


def find_breakpoints(drops, percentile=BREAKPOINT_PERCENTILE):
    """Indices after which to cut, chosen by percentile of the drops.

    Why a percentile rather than a fixed threshold: the absolute scale of
    cosine distance varies by document and by embedding model. ESMO's
    flowing prose and WHO's terse tables produce very different
    distributions, and one hard-coded number cannot suit both. A percentile
    adapts to whatever document it is handed.

    Returns:
        Sorted list of indices ``i`` meaning "cut between sentence i and
        sentence i+1".
    """
    if len(drops) == 0:
        return []
    threshold = float(np.percentile(drops, percentile))
    return [i for i, d in enumerate(drops) if d >= threshold]


def _assemble(sentences, breakpoints):
    """Group sentences into chunks, respecting size limits.

    Breakpoints are *preferences*, not commands. Three rules override them,
    in this order:

    1. Hard ceiling -- cut regardless once ``MAX_CHUNK_CHARS`` is reached,
       because text past the embedding model's token limit is silently
       dropped from the vector and would be unsearchable.
    2. Target size -- honour a breakpoint once past ``TARGET_CHUNK_CHARS``.
    3. Minimum size -- ignore a breakpoint that would emit a chunk under
       ``MIN_CHUNK_CHARS``; a two-sentence chunk rarely stands alone as
       evidence.
    """
    groups: List[List[Sentence]] = []
    current: List[Sentence] = []
    size = 0
    cuts = set(breakpoints)

    for i, sent in enumerate(sentences):
        length = len(sent.text) + 1

        # Close the current chunk BEFORE appending if this sentence would
        # push it over the ceiling. Checking after appending (the obvious
        # way to write this) lets a chunk overshoot by a whole sentence --
        # observed at 3437 chars against an 1800 limit, which silently
        # truncates inside the embedding model and makes the tail of the
        # chunk unsearchable.
        if current and size + length > MAX_CHUNK_CHARS:
            groups.append(current)
            current = []
            size = 0

        current.append(sent)
        size += length

        at_breakpoint = i in cuts
        big_enough = size >= MIN_CHUNK_CHARS
        past_target = size >= TARGET_CHUNK_CHARS

        if at_breakpoint and big_enough and past_target:
            groups.append(current)
            current = []
            size = 0

    if current:
        # Trailing remnant: merge back if it is too small to stand alone --
        # but only when the merge keeps the previous chunk under the
        # ceiling. Merging unconditionally was pushing a handful of chunks
        # past the embedding limit, which is the one thing the ceiling
        # exists to prevent.
        prev_size = sum(len(s.text) + 1 for s in groups[-1]) if groups else 0
        if groups and size < MIN_CHUNK_CHARS and prev_size + size <= MAX_CHUNK_CHARS:
            groups[-1].extend(current)
        else:
            groups.append(current)

    return groups


# ── Public entry point ─────────────────────────────────────────


def semantic_chunks(pages, paper_meta, encode, page_map=None,
                    percentile=BREAKPOINT_PERCENTILE, progress=None):
    """Chunk a document on semantic boundaries, preserving line provenance.

    Chunking is done **per page**. That is deliberate: a chunk that spans a
    page break cannot report a single coherent ``page`` + line span, and the
    citation is the product here. The cost is a few extra boundaries at page
    edges; the benefit is that every chunk has an unambiguous, verifiable
    address in the PDF.

    Args:
        pages: ``[{"text": ..., "page": physical_page}, ...]``.
        paper_meta: Per-document metadata merged into every chunk.
        encode: Callable taking ``List[str]`` and returning an ``(n, dim)``
            array. Injected rather than imported so this module never
            depends on a particular embedding library.
        page_map: Optional ``PageMap`` for printed page labels.
        percentile: Breakpoint aggressiveness; higher means fewer cuts.
        progress: Optional callable invoked as ``progress(done, total)``.

    Returns:
        ``[{"text": ..., "metadata": {...}}, ...]`` -- the same shape the
        old fixed-size chunker returned, plus line fields.
    """
    # Pass 1: sentences for the whole document, with line provenance.
    all_sentences: List[Sentence] = []
    page_spans = []  # (start_index, end_index) into all_sentences, per page
    for page in pages:
        start = len(all_sentences)
        all_sentences.extend(
            split_page_sentences(page["text"], page["page"])
        )
        page_spans.append((start, len(all_sentences)))

    if not all_sentences:
        return []

    # Pass 2: embed every sentence ONCE, in a single batched call. This is
    # the expensive step, so it must not be repeated per page.
    if progress:
        progress(0, len(all_sentences))
    vectors = np.asarray(encode([s.text for s in all_sentences]))
    if progress:
        progress(len(all_sentences), len(all_sentences))

    # Pass 3: per page, find boundaries and assemble chunks.
    chunks = []
    chunk_idx = 0
    for (start, end) in page_spans:
        sentences = all_sentences[start:end]
        if not sentences:
            continue

        drops = similarity_drops(vectors[start:end])
        breakpoints = find_breakpoints(drops, percentile)

        for group in _assemble(sentences, breakpoints):
            text = " ".join(s.text for s in group)
            if not text.strip():
                continue
            physical = group[0].page
            printed = page_map.printed_label(physical) if page_map else None
            chunks.append({
                "text": text,
                "metadata": {
                    **paper_meta,
                    "page": physical,
                    "printed_page": printed or "",
                    "chunk_index": chunk_idx,
                    # Line provenance. 1-based, relative to the page, as
                    # extracted. Display + audit aid; the viewer highlights
                    # by text search, not by these numbers.
                    "line_start": group[0].line_start,
                    "line_end": group[-1].line_end,
                    "n_sentences": len(group),
                    # Anchors for the viewer: first and last sentence let it
                    # locate the passage in the real PDF without needing the
                    # whole chunk text to match exactly.
                    "anchor_head": group[0].text[:120],
                    "anchor_tail": group[-1].text[-120:],
                },
            })
            chunk_idx += 1

    return chunks
