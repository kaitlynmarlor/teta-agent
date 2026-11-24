"""Reasoning trace: a dependency graph of what the agent did and why.

WHY THIS MODULE EXISTS
----------------------
A *log* is chronological: "then this, then that." A *trace* records
DEPENDENCY: which steps fed which conclusions. That distinction is the whole
point of this module.

Chronology cannot be filtered usefully -- drop step 3 of 7 and you leave a
hole. A dependency graph can: start at the final answer, walk backwards, and
keep only what you reach. Everything unreached is, provably, something the
answer did not depend on.

That is the definition of "filtered" here. Not "shortened", not
"summarised by an LLM" -- *reachable from the conclusion*. A reviewer can
re-run the filter and get the same result, which is what makes it auditable
rather than merely tidy.

NODE KINDS
----------
``tool_call``  A retrieval happened. Records the query, so a reviewer can
               ask "was the right question asked?"
``evidence``   A passage entered consideration, addressed by URI.
``claim``      An assertion the model made, plus the evidence it cited.

Edges run child -> parent via ``parent_ids``. A claim's parents are the
evidence nodes it cites; an evidence node's parent is the tool call that
retrieved it. So one walk from the answer recovers the full supporting
subgraph: claims, their evidence, and the searches that found it.

WHAT GETS FILTERED OUT -- AND WHY THAT IS SAFE
----------------------------------------------
Retrieval pulls 5 passages per turn; a typical answer cites 2-3. The rest
were *considered and not used*. Dropping them from the clinician view is not
hiding evidence: the full trace is retained and ``unreached()`` lists exactly
what was dropped. The clinician sees the reasoning; the auditor can still see
everything.

MAPPING TO ARIA (ARIAPlatform)
------------------------------
``TraceNode.to_aria_payload()`` emits the exact payload of
``external/ARIA/contracts/schemas/events/reasoning-node-event.schema.json``::

    {runId, nodeId, parentNodeId, decision, evidenceRefs[], uncertainty?}

``evidenceRefs`` is an array of URI strings -- precisely what
``EvidenceLedger`` already mints. This is why evidence identity had to be
built first: the filter is a reachability walk over URIs, and without stable
addresses there are no edges to walk.

Two deliberate gaps between this module and the spec, both noted inline:
ARIA allows a single ``parentNodeId`` while a claim naturally has several
parents; and ``uncertainty`` is modelled but not yet populated. See
``to_aria_payload`` for how the first is handled.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

# Node kinds. Kept as plain strings so they survive JSON round-trips.
TOOL_CALL = "tool_call"
EVIDENCE = "evidence"
CLAIM = "claim"
ANSWER = "answer"


@dataclass
class TraceNode:
    """One step in the reasoning graph.

    Attributes:
        node_id: Unique within a trace, e.g. ``"t1"``, ``"E3"``, ``"c2"``.
        kind: One of ``TOOL_CALL``, ``EVIDENCE``, ``CLAIM``, ``ANSWER``.
        summary: Human-readable one-liner. For a claim this is the sentence
            the model wrote; for a tool call, the query.
        parent_ids: Nodes this one depends on. These are the graph edges.
        evidence_uris: For claims, the URIs cited. For evidence nodes, the
            node's own URI. Empty for tool calls.
        detail: Free-form extras for rendering (scores, source names).
        turn: Which question in the conversation produced this node.
    """

    node_id: str
    kind: str
    summary: str
    parent_ids: List[str] = field(default_factory=list)
    evidence_uris: List[str] = field(default_factory=list)
    detail: Dict[str, str] = field(default_factory=dict)
    turn: int = 0

    def to_aria_payload(self, run_id):
        """Emit this node as an ARIA ``reasoning.node`` payload.

        The spec's ``parentNodeId`` is a single string, but a claim backed by
        three passages genuinely has three parents. We put the first parent in
        ``parentNodeId`` to satisfy the schema and carry the complete set in
        ``evidenceRefs``, which is already an array and is the field a
        verifier actually checks. Nothing is lost; see module docstring.
        """
        return {
            "runId": run_id,
            "nodeId": self.node_id,
            "parentNodeId": self.parent_ids[0] if self.parent_ids else "",
            "decision": self.summary,
            "evidenceRefs": list(self.evidence_uris),
        }


class ReasoningTrace:
    """Append-only graph of nodes for one case.

    Build it as the agent runs, then call ``filtered_to(answer_node_id)`` to
    get only the nodes the answer actually depended on.
    """

    def __init__(self, run_id="local"):
        self.run_id = run_id
        self._nodes: Dict[str, TraceNode] = {}
        self._order: List[str] = []

    # ── Recording ──────────────────────────────────────────────

    def add(self, node):
        """Append a node. Re-adding an existing node_id is a no-op."""
        if node.node_id not in self._nodes:
            self._nodes[node.node_id] = node
            self._order.append(node.node_id)
        return self._nodes[node.node_id]

    def record_search(self, node_id, query, turn, n_candidates=0, report=None):
        """Record that a retrieval happened."""
        detail = {"query": query}
        if report:
            detail.update(
                {
                    "considered": str(report.get("considered", "")),
                    "demoted": str(report.get("demoted", "")),
                    "sources": str(len(report.get("groups", {}))),
                }
            )
        return self.add(
            TraceNode(
                node_id=node_id,
                kind=TOOL_CALL,
                summary=query,
                detail=detail,
                turn=turn,
            )
        )

    def record_evidence(self, ref, parent_tool_id, turn):
        """Record a retrieved passage, parented to the search that found it.

        ``ref`` is an ``EvidenceRef`` from the ledger. We reuse its label as
        the node id so ``[E3]`` in the answer text and ``E3`` in the graph are
        the same identifier -- no translation table needed.
        """
        return self.add(
            TraceNode(
                node_id=ref.label,
                kind=EVIDENCE,
                summary=ref.citation(),
                parent_ids=[parent_tool_id],
                evidence_uris=[ref.uri],
                detail={
                    "source": ref.source,
                    "printed_page": ref.printed_page,
                    "similarity": f"{ref.similarity:.3f}",
                },
                turn=turn,
            )
        )

    def record_claim(self, node_id, text, cited_labels, ledger, turn):
        """Record an assertion and link it to the evidence it cites.

        Only labels that exist in the ledger become parents. A label the
        model invented has no node to point at, so it simply fails to create
        an edge -- the claim then shows up as unsupported, which is the
        correct outcome rather than an error.
        """
        parents, uris = [], []
        for label in cited_labels:
            ref = ledger.get(label)
            if ref is not None:
                parents.append(ref.label)
                uris.append(ref.uri)
        return self.add(
            TraceNode(
                node_id=node_id,
                kind=CLAIM,
                summary=text,
                parent_ids=parents,
                evidence_uris=uris,
                turn=turn,
            )
        )

    def record_answer(self, node_id, claim_ids, turn):
        """Record the final answer, parented to its constituent claims."""
        return self.add(
            TraceNode(
                node_id=node_id,
                kind=ANSWER,
                summary="Final answer",
                parent_ids=list(claim_ids),
                turn=turn,
            )
        )

    # ── Filtering ──────────────────────────────────────────────

    def ancestors_of(self, node_id):
        """Every node reachable by walking ``parent_ids`` from ``node_id``.

        Plain breadth-first search with a visited set, so cycles (which
        should not occur, but might if a caller mis-wires parents) terminate
        instead of hanging.
        """
        seen: Set[str] = set()
        queue = [node_id]
        while queue:
            current = queue.pop(0)
            if current in seen or current not in self._nodes:
                continue
            seen.add(current)
            queue.extend(self._nodes[current].parent_ids)
        return seen

    def filtered_to(self, answer_node_id):
        """Nodes the answer depends on, in original recording order.

        This is the headline operation. Note what it is NOT: no LLM, no
        heuristic, no scoring. Pure graph reachability, so two reviewers
        running it get identical output.
        """
        keep = self.ancestors_of(answer_node_id)
        return [self._nodes[n] for n in self._order if n in keep]

    def unreached(self, answer_node_id):
        """Nodes the answer did NOT depend on.

        Exposed deliberately. "Here is what was retrieved and not used" is
        itself audit-relevant -- it shows the model declined material rather
        than never seeing it.
        """
        keep = self.ancestors_of(answer_node_id)
        return [self._nodes[n] for n in self._order if n not in keep]

    # ── Access ─────────────────────────────────────────────────

    def all_nodes(self):
        return [self._nodes[n] for n in self._order]

    def get(self, node_id):
        return self._nodes.get(node_id)

    def to_aria_events(self, answer_node_id=None):
        """Emit ARIA ``reasoning.node`` payloads.

        With ``answer_node_id``, emits only the filtered subgraph -- which is
        the point: the journal carries the reasoning that mattered, not every
        candidate the retriever happened to score.
        """
        nodes = (
            self.filtered_to(answer_node_id)
            if answer_node_id
            else self.all_nodes()
        )
        return [n.to_aria_payload(self.run_id) for n in nodes]

    def __len__(self):
        return len(self._order)
