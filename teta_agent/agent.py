"""Agent orchestrator: the LLM chooses the tools.

WHY THIS MODULE EXISTS
----------------------
``orchestrator.py`` routes deterministically: a slide selects the
classifier, a question selects retrieval, both run in parallel, and the
model only writes prose at the end. That design is defensible and it is
reproducible, but it cannot express the thing a clinician actually does:

    classify the slide -> SEE that it is B2 -> *then* ask the guidelines
    about B2 specifically

The deterministic orchestrator physically cannot do this, because it
dispatches both tools concurrently -- neither result exists when the other
starts. Sequencing is the whole point of this module.

WHAT CHANGES, AND WHAT IT COSTS
-------------------------------
Here the model drives the loop: it reads the case, calls tools, reads the
results, calls more tools, and stops when it has enough. The win is a
retrieval query informed by the pathology result rather than by the user's
opening wording.

The cost is real and must not be glossed over in a demo:

    DETERMINISTIC          AGENT
    same case -> same      same case -> possibly different tool calls,
    tool calls, always     different queries, different passages

Routing stops being reproducible. The mitigation is not to pretend
otherwise but to make the *record* complete: every tool call the model
makes, with the arguments it chose, becomes a trace node. Determinism moves
from the routing to the audit of the routing.

THE GUARD RAILS
---------------
A model asking for tools is untrusted input. Three checks, all fail-closed:

1. UNKNOWN TOOL. Resolved against the registry; an invented name returns an
   error message to the model, never a dispatch.
2. INAPPLICABLE CALL. ``invoke()`` raises ``ToolNotApplicable`` for things
   like classifying a case with no slide. Returned to the model as a
   correctable error.
3. BUDGET. ``max_iterations`` and ``max_tool_calls`` bound the loop. A model
   that loops forever is stopped and the case escalates.

Note what is NOT delegated: the slide path. The model may decide *whether*
to classify, never *what file* to classify. See ``tools/adapters.py``.

ESCALATION IS STILL MECHANICAL
------------------------------
The model does not get to decide whether a human is needed. Escalations are
computed from tool output by the same rules the deterministic orchestrator
uses -- stub model, low confidence, head disagreement, empty retrieval. A
model cannot suppress one by not mentioning it.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import json
import time

from teta_agent.llm import LLMUnavailable, supports_tools
from teta_agent.orchestrator import ToolResult
from teta_agent.tools.adapters import ToolNotApplicable
from teta_agent.trace import ReasoningTrace

#: Hard ceiling on assistant turns before the loop is cut off.
MAX_ITERATIONS = 6

#: Hard ceiling on total tool executions across all turns.
MAX_TOOL_CALLS = 8

AGENT_SYSTEM_PROMPT = """You are a clinical decision-support agent for \
thymic epithelial tumors. You have tools; use them before answering.

How to work:
- If a slide is attached, classify it FIRST. Its result should shape what
  you then search for.
- Search the guidelines with specific queries that incorporate what you
  have already established (subtype, stage), not the user's raw wording.
- You may search more than once for separate sub-questions.
- Stop calling tools once you can answer, then write the answer.

Rules for the final answer:
- Cite every factual claim with its passage label, e.g. [P1].
- Never invent a label. If no passage supports a statement, say so plainly.
- Model predictions are decision SUPPORT, not diagnoses. Report the
  model's confidence and say when a human pathologist should review.
- If the pathology result and the guidelines conflict, surface the
  conflict rather than resolving it silently."""


@dataclass
class AgentStep:
    """One tool execution the model asked for.

    Attributes:
        iteration: Which assistant turn requested it, 1-based.
        tool_name: Name the model asked for, even if unknown.
        arguments: Arguments the model supplied.
        result: The ``ToolResult``, or None if the call was rejected.
        rejected: Why the call never ran. Empty when it did.
        node_id: Trace node recording this step.
    """

    iteration: int
    tool_name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    result: Optional[ToolResult] = None
    rejected: str = ""
    node_id: str = ""

    def ok(self):
        return self.result is not None and self.result.ok


@dataclass
class AgentResult:
    """Everything one agent-driven case produced."""

    case_id: str
    trace: ReasoningTrace
    steps: List[AgentStep] = field(default_factory=list)
    answer: str = ""
    iterations: int = 0
    stopped_early: bool = False
    escalations: List[str] = field(default_factory=list)
    transcript: List[Dict[str, Any]] = field(default_factory=list)
    error: str = ""
    wall_s: float = 0.0

    def tool_results(self):
        """Successful results, in execution order."""
        return [s.result for s in self.steps if s.ok()]

    def queries_asked(self):
        """Every retrieval query the MODEL composed, in order.

        The headline audit artefact of this module. In a deterministic run
        this list is just the clinician's question; here it is the agent's
        own words, and reviewing it is how you check the agent asked the
        right thing.
        """
        out = []
        for step in self.steps:
            query = step.arguments.get("query")
            if query:
                out.append(str(query))
        return out

    def needs_human(self):
        return bool(self.escalations)


class AgentOrchestrator:
    """Runs a tool-calling loop and assembles one reasoning trace.

    Args:
        tools: Adapters exposing ``name``, ``schema()``, ``invoke()``.
        llm: Provider implementing ``chat(messages, tools) -> LLMTurn``.
        max_iterations: Assistant turns before the loop is cut off.
        max_tool_calls: Total tool executions before refusing more.
        system_prompt: Override the default agent instructions.

    Raises:
        ValueError: If ``llm`` cannot do tool calling. Failing at
            construction beats discovering it mid-case.
    """

    def __init__(self, tools=None, llm=None, max_iterations=MAX_ITERATIONS,
                 max_tool_calls=MAX_TOOL_CALLS,
                 system_prompt=AGENT_SYSTEM_PROMPT):
        if llm is not None and not supports_tools(llm):
            raise ValueError(
                "{} has no chat() method, so it cannot drive an agent loop. "
                "Use Orchestrator for deterministic routing, or pass a "
                "tool-calling provider.".format(type(llm).__name__)
            )
        self.tools = list(tools or [])
        self.llm = llm
        self.max_iterations = max_iterations
        self.max_tool_calls = max_tool_calls
        self.system_prompt = system_prompt
        self._by_name = {t.name: t for t in self.tools}

    # ── Setup ──────────────────────────────────────────────────

    def schemas(self):
        """Tool schemas to advertise to the model."""
        return [t.schema() for t in self.tools if hasattr(t, "schema")]

    def opening_messages(self, request):
        """Seed the transcript.

        The slide is announced as *attached* without its path. The model
        needs to know a slide exists to decide whether to classify; it has
        no legitimate use for the filename.
        """
        facts = []
        if request.has_slide():
            facts.append(
                "A whole-slide image is attached to this case and can be "
                "classified with the pathology tool."
            )
        else:
            facts.append("No slide is attached to this case.")

        question = request.question.strip() or (
            "Summarise the pathology findings and their management "
            "implications."
        )
        user = "{}\n\n# Question\n{}".format("\n".join(facts), question)
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user},
        ]

    # ── Executing one model-requested call ─────────────────────

    def execute_call(self, request, call, iteration):
        """Run one tool call the model asked for.

        Every rejection path returns an ``AgentStep`` rather than raising:
        the model gets told what went wrong and can correct itself on the
        next turn. A bad tool call is a conversation state, not a crash.
        """
        tool = self._by_name.get(call.name)
        if tool is None:
            return AgentStep(
                iteration=iteration,
                tool_name=call.name or "(unnamed)",
                arguments=call.arguments,
                rejected="unknown tool {!r}; available: {}".format(
                    call.name, ", ".join(sorted(self._by_name)) or "none"
                ),
            )

        # Malformed JSON arrives as empty arguments. Say so explicitly --
        # otherwise a required-argument error sends the model hunting for
        # the wrong problem.
        if call.raw_arguments.strip() and not call.arguments:
            return AgentStep(
                iteration=iteration,
                tool_name=call.name,
                arguments={},
                rejected="arguments were not valid JSON; resend them as a "
                         "JSON object",
            )

        started = time.monotonic()
        try:
            result = tool.invoke(request, call.arguments)
        except ToolNotApplicable as exc:
            # Recoverable: the model asked for something invalid.
            return AgentStep(
                iteration=iteration,
                tool_name=call.name,
                arguments=call.arguments,
                rejected=str(exc),
            )
        except Exception as exc:  # noqa: BLE001 - isolate tool failures
            # Genuine failure: the tool broke. Record it as a failed result
            # so it still escalates, same as in the deterministic path.
            result = ToolResult(
                tool_name=call.name,
                ok=False,
                summary="{} failed".format(call.name),
                error="{}: {}".format(type(exc).__name__, exc),
            )
        result.elapsed_s = time.monotonic() - started
        return AgentStep(
            iteration=iteration,
            tool_name=call.name,
            arguments=call.arguments,
            result=result,
        )

    def observation_for(self, step):
        """What the model sees back from its call.

        Deliberately the same content the deterministic orchestrator puts
        in its synthesis prompt -- scope, then caveat, then evidence. The
        agent must not be shown a rosier view of retrieval than the
        deterministic path shows.
        """
        if step.rejected:
            return "ERROR: {}".format(step.rejected)
        result = step.result
        if result is None or not result.ok:
            error = result.error if result else "unknown error"
            return "TOOL FAILED: {}. Do not retry; proceed without it " \
                   "and say the result is unavailable.".format(error)

        body = result.detail.get("prompt_block") or result.summary
        scope = result.detail.get("corpus_scope")
        caveat = result.detail.get("caveat")
        parts = [p for p in (scope, caveat, body) if p]
        return "\n\n".join(parts)

    # ── Trace ──────────────────────────────────────────────────

    def record_step(self, trace, step, request):
        """Add one step to the trace and return its node id.

        The node summary is the MODEL'S query where there is one, not a
        rendering of the result. A reviewer scanning the trace should read
        the agent's own words -- that is the thing under review once the
        model is choosing what to ask.
        """
        node_id = "t{}".format(len(trace) + 1)
        query = step.arguments.get("query") or step.tool_name
        trace.record_search(node_id=node_id, query=str(query),
                            turn=request.turn)
        node = trace.get(node_id)
        if node is None:
            return node_id

        node.detail["iteration"] = str(step.iteration)
        node.detail["requested_tool"] = step.tool_name
        # The arguments the model chose, verbatim. This is the audit record
        # that replaces routing determinism.
        node.detail["model_arguments"] = json.dumps(step.arguments,
                                                    sort_keys=True)[:500]
        if step.rejected:
            node.detail["rejected"] = step.rejected
        if step.result is not None:
            node.detail.update(step.result.detail)
            node.evidence_uris = list(step.result.uris)
        step.node_id = node_id
        return node_id

    def escalations_from(self, steps, stopped_early):
        """Compute escalations from tool output, never from model prose.

        Same rules as the deterministic orchestrator: the ``escalate`` key
        a tool sets, plus outright failures. The model has no vote. A model
        that simply declines to mention low confidence cannot thereby make
        the case look safe.
        """
        out = []
        for step in steps:
            result = step.result
            if result is None:
                continue
            if not result.ok:
                out.append("{} unavailable ({}) - results incomplete".format(
                    result.tool_name, result.error))
                continue
            reason = result.detail.get("escalate")
            if reason:
                out.append("{}: {}".format(result.tool_name, reason))

        if stopped_early:
            out.append(
                "agent hit its tool/iteration budget before finishing - the "
                "answer may be incomplete"
            )
        if not any(s.ok() for s in steps):
            out.append("no tool produced a usable result - answer is ungrounded")
        return out

    # ── The loop ───────────────────────────────────────────────

    def run_case(self, request):
        """Let the model drive tools until it answers or hits the budget.

        Returns:
            AgentResult. A model outage returns a populated ``error`` with
            whatever tool results were already collected, rather than
            raising -- same degradation contract as the deterministic path.
        """
        started = time.monotonic()
        trace = ReasoningTrace(run_id=request.case_id)
        messages = self.opening_messages(request)
        schemas = self.schemas()

        steps = []
        answer = ""
        iterations = 0
        stopped_early = False
        error = ""

        while iterations < self.max_iterations:
            iterations += 1
            try:
                turn = self.llm.chat(messages, schemas)
            except LLMUnavailable as exc:
                error = str(exc)
                break

            messages.append(
                turn.raw_message
                or {"role": "assistant", "content": turn.text}
            )

            if not turn.wants_tools():
                answer = turn.text
                break

            # Budget check before executing, so an over-budget turn costs
            # nothing. The model is told, and gets one more turn to answer
            # with what it already has.
            remaining = self.max_tool_calls - len(steps)
            if remaining <= 0:
                stopped_early = True
                for call in turn.tool_calls:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call.call_id,
                        "content": "ERROR: tool-call budget exhausted. "
                                   "Answer with what you have.",
                    })
                continue

            for call in turn.tool_calls[:remaining]:
                step = self.execute_call(request, call, iterations)
                steps.append(step)
                self.record_step(trace, step, request)
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.call_id,
                    "content": self.observation_for(step),
                })
        else:
            # Loop fell through without the model ever ending its turn.
            stopped_early = True

        return AgentResult(
            case_id=request.case_id,
            trace=trace,
            steps=steps,
            answer=answer,
            iterations=iterations,
            stopped_early=stopped_early,
            escalations=self.escalations_from(steps, stopped_early),
            transcript=messages,
            error=error,
            wall_s=time.monotonic() - started,
        )

