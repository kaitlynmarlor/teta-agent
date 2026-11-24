"""Step 4: An orchestrator LLM that chooses the tools.

Contrast with ``trace_03_orchestrator.py``, which proves routing needs NO
model. This script is the other design: the model reads the case, calls the
classifier, reads the result, and *then* writes its own guidelines query.

Two modes:

    python trace_04_agent.py            offline, ScriptedLLM, no network
    python trace_04_agent.py --live     real model via argo-proxy :44497

Offline mode replays a fixed script, so the LOOP is what you are watching,
not a model. Nothing it prints is model reasoning. Use ``--live`` for that.

What the four sections show:

  1. The sequence: classify -> read result -> compose a query FROM it.
  2. The model's queries, which is the new audit surface.
  3. Guard rails: unknown tool, empty query, no-slide case, budget.
  4. The trade: same case twice, side by side with the deterministic path.
"""

import argparse
import sys
import time

from teta_agent.agent import AgentOrchestrator
from teta_agent.llm import (
    LLMTurn,
    NullLLM,
    OpenAICompatLLM,
    ScriptedLLM,
    ToolCall,
)
from teta_agent.orchestrator import CaseRequest, Orchestrator
from teta_agent.evidence import EvidenceLedger
from teta_agent.tools.adapters import GuidelinesTool, PathologyTool


# ── A fake retriever ───────────────────────────────────────────
# Stands in for Chroma + MedEmbed + reranker, which take ~30s to load.

PASSAGES = [
    ("2015-ESMO-Guidelines.pdf::0042",
     "Postoperative radiotherapy may be considered in stage II thymoma "
     "with high-risk features, including WHO type B2/B3 histology.",
     {"source": "ESMO Clinical Practice Guidelines", "author": "Girard et al.",
      "year": "2015", "page": "9", "printed_page": "v48"}, 0.785),
    ("2015-ESMO-Guidelines.pdf::0051",
     "Complete surgical resection remains the cornerstone of management "
     "for resectable thymic epithelial tumours.",
     {"source": "ESMO Clinical Practice Guidelines", "author": "Girard et al.",
      "year": "2015", "page": "7", "printed_page": "v46"}, 0.771),
]


def make_fake_search():
    """A search_fn with its own ledger, so [P] labels stay stable."""
    ledger = EvidenceLedger()

    def fake_search(query, n_results):
        time.sleep(0.1)  # pretend to embed + rerank
        return [
            ledger.add(chunk_id, text, meta, similarity=score)
            for chunk_id, text, meta, score in PASSAGES[:n_results]
        ]

    return fake_search


def banner(text):
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


# ── The offline script ─────────────────────────────────────────
# What a competent model WOULD do. Turns 2 and 3 are CALLABLES: they read
# the real tool output off the transcript, so the query names whatever
# subtype the stub actually produced. That read-then-ask dependency is the
# entire point of this module, and faking it with a hardcoded "B2" would
# demonstrate nothing.

def _assistant_call(call_id, name, arguments, raw):
    return LLMTurn(
        text="",
        tool_calls=[ToolCall(call_id=call_id, name=name,
                             arguments=arguments, raw_arguments=raw)],
        raw_message={
            "role": "assistant", "content": "",
            "tool_calls": [{"id": call_id, "type": "function",
                            "function": {"name": name, "arguments": raw}}],
        },
    )


def scripted_turns():
    def turn_1(messages):
        del messages
        return _assistant_call(
            "call_1", "pathology_classifier",
            {"reason": "A slide is attached; the subtype determines which "
                       "guidance applies."},
            '{"reason": "A slide is attached."}',
        )

    def turn_2(messages):
        # Reads turn 1's actual result and builds the query from it.
        subtype = subtype_from(last_observation(messages)) or "thymoma"
        query = ("postoperative radiotherapy indications stage II WHO type "
                 "{} thymoma".format(subtype))
        return _assistant_call(
            "call_2", "guidelines_rag", {"query": query},
            '{{"query": "{}"}}'.format(query),
        )

    def turn_3(messages):
        del messages
        text = (
            "Postoperative radiotherapy may be considered in stage II "
            "thymoma with high-risk features including B2/B3 histology "
            "[P1]. Complete resection remains the cornerstone of "
            "management [P2].\n\n"
            "IMPORTANT: the classification came from a STUB model and is "
            "not valid for clinical use; a pathologist must confirm the "
            "subtype."
        )
        return LLMTurn(text=text,
                       raw_message={"role": "assistant", "content": text})

    return [turn_1, turn_2, turn_3]


def build_tools():
    """Default stub classifier: whatever subtype it invents is fine.

    The script below reads the result instead of assuming one, so a random
    stub stays consistent with the answer.
    """
    return [
        PathologyTool(),
        GuidelinesTool(search_fn=make_fake_search()),
    ]


def last_observation(messages):
    """Text of the most recent tool message in the transcript."""
    for message in reversed(messages):
        if message.get("role") == "tool":
            return message.get("content", "")
    return ""


def subtype_from(observation):
    """Pull the six-class call out of the classifier's own prompt block.

    The stub picks a subtype at random. Parsing it back out is how the
    script reacts to what actually happened rather than to what it hoped
    would happen -- which is the behaviour being demonstrated.
    """
    for line in observation.splitlines():
        if line.startswith("WHO six-class subtype:"):
            return line.split(":", 1)[1].strip().split(" ")[0]
    return ""


def resolve_argo_user(supplied=None):
    """Get the ARGO username, from flag, env, or prompt.

    argo-proxy authenticates with your ARGO *username*, not an API key, so
    this is echoed normally rather than hidden. Mirrors
    ``rag_chat.resolve_credential``; a 403 from the proxy means this value
    is wrong, not that you need a key you do not have.
    """
    import os

    if supplied:
        return supplied
    from_env = os.environ.get("ARGO_USER", "").strip()
    if from_env:
        return from_env
    print("\nargo-proxy authenticates with your ARGO *username*, not an API")
    print("key. (Set ARGO_USER in your shell to skip this prompt.)")
    value = input("ARGO username: ").strip()
    while not value:
        value = input("Required - ARGO username: ").strip()
    return value


def make_llm(live, model=None, api_key=None):
    if not live:
        return ScriptedLLM(scripted_turns())
    kwargs = {"api_key": resolve_argo_user(api_key)}
    if model:
        kwargs["model"] = model
    return OpenAICompatLLM(**kwargs)


# ── 1. The sequence ────────────────────────────────────────────

def section_sequence(llm, request):
    banner("1. THE MODEL CHOOSES, IN SEQUENCE")

    agent = AgentOrchestrator(tools=build_tools(), llm=llm)
    result = agent.run_case(request)

    if result.error:
        print("\n  LLM unavailable: {}".format(result.error))
        print("  Run without --live for the offline script.")
        return None

    print("\n  Case: {}".format(request.case_id))
    print("  Question: {}".format(request.question))
    print("\n  What the model did, in order:\n")
    for i, step in enumerate(result.steps, start=1):
        marker = "OK " if step.ok() else "REJ"
        print("   {} [{}] iteration {} -> {}".format(
            i, marker, step.iteration, step.tool_name))
        for key, value in sorted(step.arguments.items()):
            print("        {} = {}".format(key, str(value)[:90]))
        if step.rejected:
            print("        rejected: {}".format(step.rejected))
        elif step.result is not None:
            print("        -> {}".format(step.result.summary[:90]))

    print("\n  Assistant turns: {}   tool calls: {}   wall: {:.2f}s".format(
        result.iterations, len(result.steps), result.wall_s))
    print("\n  Final answer:\n")
    for line in (result.answer or "(none)").splitlines():
        print("    {}".format(line))
    return result


# ── 2. The new audit surface ───────────────────────────────────

def section_queries(result):
    banner("2. THE MODEL'S QUERIES ARE THE NEW AUDIT SURFACE")

    queries = result.queries_asked()
    print("\n  Queries the MODEL composed (nobody typed these):")
    for i, query in enumerate(queries, start=1):
        print("    {}. {}".format(i, query))
    if not queries:
        print("    (none)")

    print("\n  Why this matters: the deterministic orchestrator searches the")
    print("  clinician's words verbatim. Here the model wrote the query, so")
    print("  'was the right question asked?' is now a real review step.")

    print("\n  Trace nodes:")
    for node in result.trace.all_nodes():
        author = node.detail.get("query_author", "-")
        print("    {} [{}] author={} | {}".format(
            node.node_id, node.kind, author, node.summary[:52]))

    first = result.trace.get("t1")
    if first is not None:
        print("\n  Node t1 model_arguments (verbatim):")
        print("    {}".format(first.detail.get("model_arguments", "-")))
        print("\n  ARIA payload for t1:")
        payload = first.to_aria_payload(result.case_id)
        print("    nodeId      : {}".format(payload["nodeId"]))
        print("    decision    : {}".format(payload["decision"][:58]))
        print("    evidenceRefs: {} ...".format(payload["evidenceRefs"][:2]))

    print("\n  Escalations (computed from tool output, NOT from the model):")
    for item in result.escalations:
        print("    ! {}".format(item))
    print("  needs_human(): {}".format(result.needs_human()))


# ── 3. Guard rails ─────────────────────────────────────────────

def section_guards():
    banner("3. GUARD RAILS: THE MODEL'S REQUESTS ARE UNTRUSTED INPUT")

    request = CaseRequest(question="What does ESMO say?",
                          slide_path="/slides/UC-0042.svs",
                          case_id="UC-0042")

    def turn(call_id, name, args_dict, raw):
        return LLMTurn(
            text="",
            tool_calls=[ToolCall(call_id=call_id, name=name,
                                 arguments=args_dict, raw_arguments=raw)],
            raw_message={"role": "assistant", "content": ""},
        )

    # a. Unknown tool.
    agent = AgentOrchestrator(
        tools=build_tools(),
        llm=ScriptedLLM([
            turn("c1", "order_biopsy", {"site": "mediastinum"},
                 '{"site": "mediastinum"}'),
            LLMTurn(text="Understood, that tool does not exist.",
                    raw_message={"role": "assistant", "content": "ok"}),
        ]),
    )
    result = agent.run_case(request)
    print("\n  a. Model invents a tool:")
    print("     rejected -> {}".format(result.steps[0].rejected))

    # b. Empty query.
    agent = AgentOrchestrator(
        tools=build_tools(),
        llm=ScriptedLLM([
            turn("c1", "guidelines_rag", {"query": "   "}, '{"query": "   "}'),
            LLMTurn(text="Retrying with a real query.",
                    raw_message={"role": "assistant", "content": "ok"}),
        ]),
    )
    result = agent.run_case(request)
    print("\n  b. Model sends an empty query:")
    print("     rejected -> {}".format(result.steps[0].rejected))

    # c. Classify a case with no slide.
    agent = AgentOrchestrator(
        tools=build_tools(),
        llm=ScriptedLLM([
            turn("c1", "pathology_classifier", {"reason": "curiosity"},
                 '{"reason": "curiosity"}'),
            LLMTurn(text="No slide, so I cannot classify.",
                    raw_message={"role": "assistant", "content": "ok"}),
        ]),
    )
    result = agent.run_case(
        CaseRequest(question="What is a B3 thymoma?", case_id="no-slide"))
    print("\n  c. Model classifies a case with no slide:")
    print("     rejected -> {}".format(result.steps[0].rejected))

    # d. Budget.
    loop = [turn("c{}".format(i), "guidelines_rag",
                 {"query": "thymoma question {}".format(i)},
                 '{"query": "q"}') for i in range(10)]
    agent = AgentOrchestrator(tools=build_tools(), llm=ScriptedLLM(loop),
                              max_iterations=4, max_tool_calls=2)
    result = agent.run_case(request)
    print("\n  d. Model loops forever (budget: 4 turns, 2 calls):")
    print("     tool calls executed: {}".format(len(result.steps)))
    print("     stopped_early      : {}".format(result.stopped_early))
    print("     escalations        : {}".format(len(result.escalations)))

    # e. The slide path was never offered to the model.
    agent = AgentOrchestrator(tools=build_tools(), llm=ScriptedLLM([]))
    messages = agent.opening_messages(request)
    leaked = request.slide_path in messages[1]["content"]
    print("\n  e. Is the slide path exposed to the model? {}".format(
        "YES - BUG" if leaked else "no"))
    print("     The model decides WHETHER to classify, never WHAT file.")


# ── 4. The trade ───────────────────────────────────────────────

def section_tradeoff(request):
    banner("4. THE TRADE: DETERMINISTIC vs AGENT, SAME CASE")

    # Deterministic path. NullLLM proves routing consults no model.
    det = Orchestrator(tools=build_tools(), llm=NullLLM())
    det_result = det.run_case(request, synthesize=False)
    det_node = det_result.trace.get("t2")

    print("\n  DETERMINISTIC (orchestrator.py)")
    print("    routing model      : none (NullLLM never raised)")
    print("    tools              : parallel, both at once")
    print("    guidelines query   : {!r}".format(
        det_node.detail.get("query", "") if det_node else ""))
    print("    query author       : {}".format(
        det_node.detail.get("query_author", "-") if det_node else "-"))
    print("    same input -> same tool calls, always")

    agent = AgentOrchestrator(tools=build_tools(),
                              llm=ScriptedLLM(scripted_turns()))
    agent_result = agent.run_case(request)
    agent_queries = agent_result.queries_asked()

    print("\n  AGENT (agent.py)")
    print("    routing model      : yes, chooses every call")
    print("    tools              : sequential, each informed by the last")
    print("    guidelines query   : {!r}".format(
        agent_queries[0] if agent_queries else ""))
    print("    query author       : model")
    print("    same input -> NOT guaranteed to repeat")

    # Name the subtype the stub actually produced, rather than asserting
    # one. The whole claim is that the query depends on the tool result, so
    # the narration has to depend on it too.
    pathology = next(
        (s.result for s in agent_result.steps
         if s.ok() and s.tool_name == "pathology_classifier"), None)
    subtype = pathology.detail.get("subtype", "?") if pathology else "?"

    print("""
  The agent query names {0}. That token exists only because the classifier
  ran first and the model read its output -- the stub picks a subtype at
  random, and the query follows whatever it picked. The deterministic path
  cannot produce this: both tools start before either finishes.

  What you give up is routing reproducibility. What replaces it is a
  complete record: every call, with the arguments the model chose, in the
  trace. Determinism moves from the routing to the audit of the routing.

  Escalation does NOT move. Both paths compute it from tool output, so a
  model cannot hide a low-confidence call by declining to mention it.
""".format(subtype))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true",
                        help="use a real model via argo-proxy instead of "
                             "the offline script")
    parser.add_argument("--model", default=None,
                        help="override the model id (implies --live)")
    parser.add_argument("--argo-user", default=None,
                        help="ARGO username; falls back to $ARGO_USER, "
                             "then prompts")
    args = parser.parse_args()

    live = args.live or bool(args.model)
    llm = make_llm(live, model=args.model, api_key=args.argo_user)

    print("=" * 70)
    print("TETA agent orchestrator -- {}".format(
        "LIVE: {}".format(llm.name) if live else "OFFLINE (ScriptedLLM)"))
    print("=" * 70)
    if not live:
        print("\nThe offline script replays canned turns. It demonstrates the")
        print("LOOP, not a model. Nothing below is model reasoning.")
        print("Use --live for that.")

    request = CaseRequest(
        question="Is postoperative radiotherapy indicated for this patient?",
        slide_path="/slides/UC-0042.svs",
        case_id="UC-0042",
    )

    result = section_sequence(llm, request)
    if result is None:
        return 1
    section_queries(result)
    section_guards()
    section_tradeoff(request)

    print("\nEvery pathology number above is fabricated by the stub.")
    print("Swap in Sacco weights via PathologyTool(classify_fn=...).\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

