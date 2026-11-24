"""Turn answer prose into claim nodes, and the filtered graph into a report.

WHY THIS MODULE EXISTS
----------------------
``trace.py`` defines the graph but cannot build the claim layer on its own:
something has to read the model's prose and work out *which sentence cited
which evidence*. That is this module's first job.

The approach is deliberately mechanical -- a regex for ``[P3]`` and sentence
splitting -- not a second LLM call. Using a model to summarise a model's
reasoning would make the trace an interpretation rather than a record. If the
extraction is wrong here, it is wrong in a way a reviewer can see and
reproduce, which is the property that matters.

THE "AS CITED IN" CASE
----------------------
The system prompt asks for second-hand claims in the form::

    "Loehrer et al. reported 57% (as cited in [P3])"

``extract_claims`` detects that phrasing and marks the claim ``second_hand``,
so the renderer can show the clinician that the number came through an
intermediary rather than from a paper we hold. Collapsing this distinction is
exactly what the old ``[Source N]`` notation did wrong.

UNSUPPORTED CLAIMS
------------------
A sentence that makes an assertion and cites nothing is reported, not
discarded. In a clinical setting an uncited assertion is the single most
important thing for a reviewer to notice, so suppressing it would defeat the
purpose of the trace.

PORTABILITY CONTRACT (ARIAPlatform)
-----------------------------------
``extract_claims`` is domain-neutral: it knows about ``[P_n]`` notation, not
about thymoma. ``render_clinician_report`` is the only place with any
presentation opinion, and it reads solely from ``TraceNode`` fields. When
this graduates, the renderer becomes one of several views over the same
filtered graph (clinician, auditor, debugger).
"""

import re
from typing import List, Optional

from teta_agent.evidence import LABEL_PREFIX
from teta_agent.trace import CLAIM, EVIDENCE, TOOL_CALL

# Matches [P1], [P12]. Deliberately strict: ordinary bracketed text in these
# guidelines includes Vancouver reference numbers ("[50]") and ESMO grading
# ("[II, E]"), and neither must be mistaken for a passage label. Built from
# LABEL_PREFIX so a future rename stays a one-line change.
_LABEL = re.compile(r"\[(" + LABEL_PREFIX + r"\d+)\]")

# "(as cited in [P3])" / "as cited in [P3]" -- the second-hand marker.
_AS_CITED_IN = re.compile(
    r"\(?\bas cited in\s+((?:\[" + LABEL_PREFIX + r"\d+\][,\s and]*)+)\)?",
    re.IGNORECASE,
)

# Markdown scaffolding that is not a claim: headings, table rules, bullets
# that only introduce a section.
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s")
_TABLE_RULE = re.compile(r"^\s*\|?[\s:|-]+\|[\s:|-]*$")


def _looks_like_prose(line):
    """Filter out markdown chrome before claim extraction."""
    stripped = line.strip()
    if not stripped:
        return False
    if _HEADING.match(stripped) or _TABLE_RULE.match(stripped):
        return False
    if set(stripped) <= set("-—=*_ "):
        return False
    return True


def split_sentences(text):
    """Split prose into sentence-ish units.

    Two subtleties, both learned the hard way:

    1. **Unwrap before splitting.** Markdown prose is hard-wrapped, so a
       single sentence spans several lines. Splitting line-by-line yields
       fragments like ``"grade [E3]."`` -- which tells a clinician nothing
       and makes the claim node useless. So we first join wrapped lines back
       into logical blocks, then split those into sentences.

    2. **Do not split on every period.** ``"et al."`` and ``"p.v48"`` would
       shatter. We split only on sentence-ending punctuation followed by
       whitespace and a capital letter or bracket.

    Bullets and table rows are treated as atomic units -- they are already
    one assertion each, and sentence-splitting them adds nothing.
    """
    # Pass 1: gather lines into logical blocks, unwrapping soft line breaks.
    blocks: List[str] = []
    buffer: List[str] = []

    def flush():
        if buffer:
            blocks.append(" ".join(buffer))
            buffer.clear()

    for line in text.split("\n"):
        stripped = line.strip()
        if not _looks_like_prose(line):
            flush()
            continue
        # A bullet, numbered item, or table row starts a new block.
        if stripped.startswith("|") or re.match(r"^[-*+]\s|^\d+\.\s", stripped):
            flush()
            blocks.append(stripped)
            continue
        buffer.append(stripped)
    flush()

    # Pass 2: split each block into sentences.
    units: List[str] = []
    for block in blocks:
        if block.startswith("|"):
            units.append(block)
            continue
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z(\[])", block)
        units.extend(p.strip() for p in parts if p.strip())
    return units


class ExtractedClaim:
    """One assertion from the answer, with the labels it cited.

    Attributes:
        text: The sentence as written.
        labels: Evidence labels cited, in order of appearance.
        second_hand_labels: Labels appearing inside an "as cited in" phrase.
        supported: Whether any label was cited at all.
    """

    def __init__(self, text, labels, second_hand_labels):
        self.text = text
        self.labels = labels
        self.second_hand_labels = second_hand_labels
        self.supported = bool(labels)

    @property
    def is_second_hand(self):
        return bool(self.second_hand_labels)

    def __repr__(self):
        return f"<ExtractedClaim {self.labels} {self.text[:40]!r}>"


def extract_claims(answer_text, known_labels=None):
    """Pull cited assertions out of the model's answer.

    Args:
        answer_text: The model's full response.
        known_labels: Optional set of labels the ledger actually allocated.
            Labels outside this set are recorded as ``hallucinated`` on the
            returned tuple rather than silently dropped.

    Returns:
        ``(claims, hallucinated)`` -- a list of ``ExtractedClaim`` for every
        sentence that cites at least one label, plus a sorted list of any
        cited labels that do not exist in the ledger.
    """
    claims: List[ExtractedClaim] = []
    hallucinated = set()

    for sentence in split_sentences(answer_text):
        labels = _LABEL.findall(sentence)
        if not labels:
            continue

        second_hand = []
        for match in _AS_CITED_IN.finditer(sentence):
            second_hand.extend(_LABEL.findall(match.group(1)))

        if known_labels is not None:
            for label in labels:
                if label not in known_labels:
                    hallucinated.add(label)
            labels = [lab for lab in labels if lab in known_labels]
            second_hand = [lab for lab in second_hand if lab in known_labels]

        if labels:
            # De-duplicate while preserving order.
            seen = set()
            ordered = [x for x in labels if not (x in seen or seen.add(x))]
            claims.append(ExtractedClaim(sentence, ordered, second_hand))

    return claims, sorted(hallucinated)


def populate_trace(trace, answer_text, ledger, turn, answer_node_id="answer"):
    """Add claim nodes and an answer node to a trace.

    Returns ``(answer_node_id, claims, hallucinated, second_hand_ids)``.
    ``second_hand_ids`` is the set of claim node ids whose assertion came
    through an "as cited in" attribution; the renderer flags these.
    """
    claims, hallucinated = extract_claims(
        answer_text, known_labels=ledger.known_labels()
    )
    claim_ids = []
    second_hand_ids = set()
    for i, claim in enumerate(claims, start=1):
        node_id = f"c{turn}.{i}"
        trace.record_claim(node_id, claim.text, claim.labels, ledger, turn)
        claim_ids.append(node_id)
        if claim.is_second_hand:
            second_hand_ids.add(node_id)

    trace.record_answer(answer_node_id, claim_ids, turn)
    return answer_node_id, claims, hallucinated, second_hand_ids


# ── Rendering ──────────────────────────────────────────────────


def render_clinician_report(trace, answer_node_id, ledger=None,
                            second_hand_ids=None, only_turn=None):
    """Render the filtered trace so each claim sits with its own evidence.

    The grouping is the requirement: a clinician should not have to
    cross-reference a numbered list at the bottom of the page. Each claim is
    printed with the passages that support it, and each passage with the
    search that found it.

    Args:
        second_hand_ids: Claim node ids whose assertion was attributed via
            "as cited in". Surfaced explicitly because "the review says X"
            and "X" carry different evidential weight, and collapsing them
            is the failure the whole evidence layer exists to prevent.
        only_turn: Scope the "considered and not used" section to one turn.
            Required for multi-turn sessions: ``trace.unreached`` is
            correctly session-wide, so without this, question 2's report
            would list every node from question 1 as "not used" -- true of
            *that* answer, but grossly misleading to a reader who is looking
            at one question. The kept/claim side needs no scoping, since a
            turn's claims are only ever ancestors of that turn's answer.
    """
    second_hand_ids = set(second_hand_ids or ())
    kept = trace.filtered_to(answer_node_id)
    dropped = trace.unreached(answer_node_id)
    if only_turn is not None:
        dropped = [n for n in dropped if n.turn == only_turn]

    by_id = {n.node_id: n for n in trace.all_nodes()}
    claims = [n for n in kept if n.kind == CLAIM]

    lines: List[str] = []
    add = lines.append

    add("=" * 70)
    add("CLINICAL REASONING TRACE")
    add("=" * 70)
    add("")
    add(f"{len(claims)} supported claims led to this answer.")
    add("Each is shown with the evidence it rests on and how that evidence")
    add("was found.")
    add("")

    for i, claim in enumerate(claims, start=1):
        add("-" * 70)
        tag = "  —  SECOND-HAND" if claim.node_id in second_hand_ids else ""
        add(f"CLAIM {i}{tag}")
        add(f"  {claim.summary}")
        if claim.node_id in second_hand_ids:
            add("  ⚠️  Attributed to a study we did NOT retrieve. The figure")
            add("      comes via the passage below, not from the original.")
        add("")

        if not claim.parent_ids:
            add("  ⚠️  NO EVIDENCE CITED — this assertion is unsupported.")
            add("")
            continue

        add("  Rests on:")
        for label in claim.parent_ids:
            ev = by_id.get(label)
            if ev is None or ev.kind != EVIDENCE:
                continue
            add(f"    [{label}] {ev.summary}")
            add(f"          similarity {ev.detail.get('similarity', '?')}"
                f"   {ev.evidence_uris[0] if ev.evidence_uris else ''}")

            # Walk one more hop: which search surfaced this passage?
            for parent in ev.parent_ids:
                tool = by_id.get(parent)
                if tool is not None and tool.kind == TOOL_CALL:
                    add(f"          found by: \"{tool.summary}\"")
        add("")

    add("=" * 70)
    add("WHAT WAS CONSIDERED AND NOT USED")
    add("=" * 70)
    if not dropped:
        add("  (nothing — every retrieved passage was cited)")
    else:
        for node in dropped:
            if node.kind == EVIDENCE:
                add(f"  [{node.node_id}] {node.summary}")
            elif node.kind == TOOL_CALL:
                add(f"  search: \"{node.summary}\" — produced nothing cited")
    add("")
    add("  These were retrieved but the answer does not depend on them.")
    add("  Listed so the record shows what was available, not just what")
    add("  was used.")

    return "\n".join(lines)


# ── Consolidated views ─────────────────────────────────────────
#
# WHY A SECOND RENDERER RATHER THAN A SHORTER FIRST ONE
#
# render_clinician_report repeats an evidence block under every claim that
# cites it. For one question that is a feature -- the clinician reads a
# claim and its support together, with no cross-referencing. But the cost
# is quadratic-ish in practice: five claims resting on two shared passages
# printed eight evidence blocks and repeated "found by:" six times in a
# measured run.
#
# Across a whole case -- many questions, many tools -- that stops being
# readable. The fix is NOT to trim the detailed view; an auditor chasing one
# specific claim still wants it. The fix is a different *level of zoom* over
# the same filtered graph. Summary to triage, detail to investigate.
#
# Both read from the identical node set, so they cannot disagree. That is
# the property that matters: a summary that could drift from its detail
# view is worse than no summary.


def _evidence_nodes(trace, answer_ids):
    """Evidence nodes supporting any of ``answer_ids``, deduplicated.

    Returns ``{node_id: (node, [claim nodes that cite it])}`` in first-seen
    order.
    """
    out = {}
    for answer_id in answer_ids:
        kept = trace.filtered_to(answer_id)
        by_id = {n.node_id: n for n in kept}
        for node in kept:
            if node.kind != CLAIM:
                continue
            for label in node.parent_ids:
                ev = by_id.get(label)
                if ev is None or ev.kind != EVIDENCE:
                    continue
                if label not in out:
                    out[label] = (ev, [])
                out[label][1].append(node)
    return out


def render_case_summary(trace, answers, ledger=None):
    """One-screen overview of an entire case, however many turns it ran.

    Args:
        trace: The session's ``ReasoningTrace``.
        answers: Ordered list of ``{"answer_id", "turn", "question",
            "second_hand"}`` dicts -- one per answered question.
        ledger: Unused today; accepted so the signature matches the detailed
            renderer and callers can swap between them.

    The ordering principle is **exceptions first**. A clinician auditing a
    long case does not read top to bottom; they look for what needs
    challenging. So unsupported claims, fabricated citations and second-hand
    figures are surfaced above the routine per-question breakdown, rather
    than being left for the reader to notice.
    """
    answer_ids = [a["answer_id"] for a in answers]
    all_second_hand = set()
    for a in answers:
        all_second_hand |= set(a.get("second_hand") or ())

    ev_map = _evidence_nodes(trace, answer_ids)

    claims = []
    for answer_id in answer_ids:
        claims.extend(
            n for n in trace.filtered_to(answer_id) if n.kind == CLAIM
        )

    unsupported = [c for c in claims if not c.parent_ids]
    second_hand = [c for c in claims if c.node_id in all_second_hand]

    # Group evidence by source document -- "4 passages from ESMO" is the
    # unit a clinician thinks in, not four separate page references.
    by_source = {}
    for node, citing in ev_map.values():
        src = node.detail.get("source", "Unknown")
        by_source.setdefault(src, []).append((node, citing))

    lines: List[str] = []
    add = lines.append

    add("=" * 70)
    add("CASE SUMMARY")
    add("=" * 70)
    add("")
    add(f"  {len(answers)} questions   "
        f"{len(claims)} supported claims   "
        f"{len(ev_map)} distinct passages   "
        f"{len(by_source)} sources")
    add("")

    # ── Exceptions, first and unmissable ──────────────────────
    flagged = bool(unsupported or second_hand)
    if flagged:
        add("-" * 70)
        add("NEEDS REVIEW")
        add("-" * 70)
        for claim in unsupported:
            add(f"  ⚠️  UNSUPPORTED  (Q{claim.turn})  {claim.summary[:52]}")
        for claim in second_hand:
            add(f"  ⚠️  SECOND-HAND  (Q{claim.turn})  {claim.summary[:52]}")
        add("")
    else:
        add("  ✅ No unsupported or second-hand claims.")
        add("")

    # ── Per-question, one line each ───────────────────────────
    add("-" * 70)
    add("BY QUESTION")
    add("-" * 70)
    for entry in answers:
        answer_id = entry["answer_id"]
        kept = trace.filtered_to(answer_id)
        q_claims = [n for n in kept if n.kind == CLAIM]
        q_ev = sorted({
            label
            for c in q_claims
            for label in c.parent_ids
            if trace.get(label) is not None
            and trace.get(label).kind == EVIDENCE
        })
        question = entry.get("question", "")
        add(f"  Q{entry['turn']}: {question[:58]}")
        add(f"       {len(q_claims)} claims  ←  {', '.join(q_ev) or 'none'}")
    add("")

    # ── Evidence base, grouped by document ────────────────────
    add("-" * 70)
    add("EVIDENCE BASE")
    add("-" * 70)
    for src in sorted(by_source):
        entries = by_source[src]
        add(f"  {src}  ({len(entries)} "
            f"passage{'' if len(entries) == 1 else 's'})")
        for node, citing in sorted(entries, key=lambda x: x[0].node_id):
            turns = sorted({c.turn for c in citing})
            where = ", ".join(f"Q{t}" for t in turns)
            page = node.detail.get("printed_page") or "?"
            n_cited = len(citing)
            add(f"    [{node.node_id}] p.{page:<6s} "
                f"cited by {n_cited} claim{'' if n_cited == 1 else 's'} "
                f"in {where}")
    add("")
    add("  Each passage appears once here regardless of how many claims")
    add("  rest on it. Use the detailed trace to see claim-by-claim support.")

    return "\n".join(lines)
