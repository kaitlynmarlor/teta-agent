"""Central orchestrator: dispatch tools, merge results, build one trace.

WHY THIS MODULE EXISTS
----------------------
Two capabilities now exist independently -- guideline retrieval and the
pathology classifier -- and a real clinical question needs both:

    "This slide looks like B2. What does ESMO say about managing it?"

Answering that is two calls whose results must land in ONE auditable trace.
This module is that join.

THE CENTRAL DESIGN CLAIM: ROUTING NEEDS NO LLM
----------------------------------------------
It is tempting to let a model decide which tools to call. We do not, and the
split is deliberate:

    ROUTING    (which tools, what order, escalate?)  -> deterministic Python
    SYNTHESIS  (prose with [P1] citations)           -> the LLM

Routing is policy. Policy must be reproducible, testable, and reviewable --
rerun the same case, get the same tool calls. An LLM there would make the
audit trail itself nondeterministic, which defeats the point of having one.
``plan()`` is therefore a pure function, and the test suite passes
``NullLLM`` to *prove* no model is consulted before synthesis.

A practical consequence: the whole orchestration layer runs on CPU, offline,
in milliseconds. Only ``synthesize()`` touches a network.

PARALLELISM
-----------
The two tools are independent, so they run concurrently on a thread pool.
Threads, not processes: both tools are I/O- or C-extension-bound (HTTP,
ChromaDB, torch), all of which release the GIL. Order of completion is
nondeterministic, so results are re-sorted into plan order before merging --
otherwise the trace would differ run to run and break reproducibility.

SWAPPING IN THE REAL COMPONENTS
-------------------------------
Nothing here imports the stub, ChromaDB, or ``openai``. Tools are injected
and satisfy one tiny protocol (``name``, ``run(request)``)::

    Orchestrator(tools=[PathologyTool(classify_fn=real_slideflow_predict),
                        GuidelinesTool(search_fn=chroma_search)],
                 llm=OpenAICompatLLM(model="argo:gpt-5"))

Replacing the stub with Sacco weights means passing a different
``classify_fn``. See ``tools/adapters.py``.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import time

from teta_agent.llm import LLMUnavailable
from teta_agent.trace import ReasoningTrace

#: Confidence below which a pathology call is escalated rather than trusted.
REVIEW_THRESHOLD = 0.70


@dataclass
class CaseRequest:
    """One clinical question, optionally with a slide attached.

    Attributes:
        question: Free-text clinical question. May be empty when the case is
            slide-only (classify this, nothing asked).
        slide_path: Path to a WSI, or None for a guidelines-only question.
        case_id: Identifier used as the trace run id.
        turn: Conversation turn, threaded into every trace node.
    """

    question: str = ""
    slide_path: Optional[str] = None
    case_id: str = "case-local"
    turn: int = 1

    def has_slide(self):
        return bool(self.slide_path)

    def has_question(self):
        return bool(self.question.strip())


@dataclass
class ToolResult:
    """Uniform envelope returned by every tool.

    Making this uniform is what lets the orchestrator merge heterogeneous
    tools without knowing what any of them do.

    Attributes:
        tool_name: Stable name, used as the trace node id prefix.
        ok: False when the tool raised. A failed tool must not abort the
            case -- a pathology outage should still allow a guidelines
            answer, so failures are recorded and routed around.
        summary: One-line human-facing result.
        uris: Addresses of whatever the tool looked at. Drops straight into
            an ARIA ``evidenceRefs`` array.
        detail: Stringified fields for ``TraceNode.detail``.
        payload: The rich native object (PathologyPrediction, list of
            EvidenceRef, ...). Never written to the trace; for callers.
        error: Message when ``ok`` is False.
        elapsed_s: Wall-clock duration, for the parallelism claim.
    """

    tool_name: str
    ok: bool = True
    summary: str = ""
    uris: List[str] = field(default_factory=list)
    detail: Dict[str, str] = field(default_factory=dict)
    payload: Any = None
    error: str = ""
    elapsed_s: float = 0.0


@dataclass
class CaseResult:
    """Everything one orchestrated case produced."""

    case_id: str
    trace: ReasoningTrace
    tool_results: List[ToolResult] = field(default_factory=list)
    answer: str = ""
    synthesis_ok: bool = True
    synthesis_error: str = ""
    escalations: List[str] = field(default_factory=list)
    wall_s: float = 0.0

    def by_tool(self, name):
        """First result from ``name``, or None."""
        for result in self.tool_results:
            if result.tool_name == name:
                return result
        return None

    def failed_tools(self):
        return [r.tool_name for r in self.tool_results if not r.ok]

    def needs_human(self):
        """True when anything demands a human before clinical use."""
        return bool(self.escalations)


SYNTHESIS_SYSTEM_PROMPT = """You are a clinical decision-support assistant \
for thymic epithelial tumors.

Rules:
- Cite every factual claim with its passage label, e.g. [P1].
- Never invent a label. If no passage supports a statement, say so plainly.
- Model predictions are decision SUPPORT, not diagnoses. Report the model's
  confidence and say when a human pathologist should review.
- If the pathology result and the guidelines conflict, surface the conflict
  rather than resolving it silently."""


class Orchestrator:
    """Runs tools for a case and assembles one reasoning trace.

    Args:
        tools: Objects satisfying ``name`` + ``run(CaseRequest) -> ToolResult``.
        llm: Synthesis provider with ``.complete(system, user)``. Only used
            by ``synthesize()``; routing never touches it.
        max_workers: Thread-pool width for parallel dispatch.
        review_threshold: Confidence floor before escalation.
    """

    def __init__(self, tools=None, llm=None, max_workers=4,
                 review_threshold=REVIEW_THRESHOLD):
        self.tools = list(tools or [])
        self.llm = llm
        self.max_workers = max_workers
        self.review_threshold = review_threshold

    # ── 1. Plan ────────────────────────────────────────────────

    def plan(self, request):
        """Choose which tools to run. Pure, deterministic, model-free.

        A tool is selected when the request carries the input it needs:
        a slide selects the classifier, a question selects guidelines. The
        rule is intentionally dull -- it must be restatable to a reviewer in
        one sentence.

        Returns:
            List of tools, in a stable order.
        """
        selected = []
        for tool in self.tools:
            if tool.applies_to(request):
                selected.append(tool)
        return selected

    # ── 2. Dispatch ────────────────────────────────────────────

    def dispatch(self, request, tools):
        """Run tools concurrently; return results in PLAN order.

        Completion order is nondeterministic, so we index results back to
        plan position. Without this the trace node order -- and therefore
        the audit artefact -- would vary between identical runs.

        A tool that raises yields ``ok=False`` instead of propagating: one
        broken capability must not sink the case.
        """
        if not tools:
            return []

        results = [None] * len(tools)

        def _run(index, tool):
            started = time.monotonic()
            try:
                result = tool.run(request)
            except Exception as exc:  # noqa: BLE001 - isolate tool failures
                result = ToolResult(
                    tool_name=tool.name,
                    ok=False,
                    summary="{} failed".format(tool.name),
                    error="{}: {}".format(type(exc).__name__, exc),
                )
            result.elapsed_s = time.monotonic() - started
            return index, result

        workers = min(self.max_workers, len(tools))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for index, result in pool.map(
                lambda pair: _run(*pair), list(enumerate(tools))
            ):
                results[index] = result
        return results

    # ── 3. Merge ───────────────────────────────────────────────

    def merge(self, request, results):
        """Fold tool results into one trace and collect escalations.

        Each tool becomes a ``tool_call`` node carrying its URIs. Evidence
        nodes are added by the tools themselves where they have an
        ``EvidenceLedger``; the orchestrator stays domain-agnostic.
        """
        trace = ReasoningTrace(run_id=request.case_id)
        escalations = []

        for position, result in enumerate(results, start=1):
            node_id = "t{}".format(position)
            trace.record_search(
                node_id=node_id,
                query=result.summary or result.tool_name,
                turn=request.turn,
            )
            node = trace.get(node_id)
            if node is not None:
                node.detail.update(result.detail)
                node.evidence_uris = list(result.uris)

            if not result.ok:
                escalations.append(
                    "{} unavailable ({}) - results incomplete".format(
                        result.tool_name, result.error
                    )
                )
                continue

            reason = result.detail.get("escalate")
            if reason:
                escalations.append("{}: {}".format(result.tool_name, reason))

        return trace, escalations

    # ── 4. Synthesize ──────────────────────────────────────────

    def build_prompt(self, request, results):
        """Assemble the synthesis prompt from successful tool results."""
        blocks = []
        for result in results:
            if not result.ok:
                blocks.append(
                    "## {}\n(unavailable: {})".format(
                        result.tool_name, result.error
                    )
                )
                continue
            # Scope, then caveat, then evidence: the model should read what
            # the corpus cannot answer *before* the passages, not after.
            # Both keys are optional -- tools that set neither are unchanged.
            body = result.detail.get("prompt_block") or result.summary
            scope = result.detail.get("corpus_scope")
            caveat = result.detail.get("caveat")
            parts = [p for p in (scope, caveat, body) if p]
            blocks.append(
                "## {}\n{}".format(result.tool_name, "\n\n".join(parts))
            )

        question = request.question.strip() or (
            "Summarise the pathology findings and their management "
            "implications."
        )
        return "{}\n\n# Question\n{}".format("\n\n".join(blocks), question)

    def synthesize(self, request, results):
        """Generate the prose answer. The ONLY method that needs a model.

        Returns:
            Tuple ``(answer, ok, error)``. On failure the error is returned
            rather than raised, so tool results and the trace survive a
            model outage.
        """
        if self.llm is None:
            return "", False, "no LLM configured"
        prompt = self.build_prompt(request, results)
        try:
            answer = self.llm.complete(SYNTHESIS_SYSTEM_PROMPT, prompt)
        except LLMUnavailable as exc:
            return "", False, str(exc)
        return answer, True, ""

    # ── Full pipeline ──────────────────────────────────────────

    def run_case(self, request, synthesize=True):
        """Plan -> dispatch -> merge -> (optionally) synthesize.

        Args:
            request: The case.
            synthesize: When False, no model is contacted at all. This is
                the switch that makes the whole pipeline runnable offline.

        Returns:
            CaseResult.
        """
        started = time.monotonic()
        tools = self.plan(request)
        results = self.dispatch(request, tools)
        trace, escalations = self.merge(request, results)

        answer, ok, error = "", True, ""
        if synthesize:
            answer, ok, error = self.synthesize(request, results)
            if not ok:
                escalations.append("synthesis unavailable: {}".format(error))

        return CaseResult(
            case_id=request.case_id,
            trace=trace,
            tool_results=results,
            answer=answer,
            synthesis_ok=ok,
            synthesis_error=error,
            escalations=escalations,
            wall_s=time.monotonic() - started,
        )
