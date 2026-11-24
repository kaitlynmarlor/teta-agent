"""Step 3: Two tools, one trace -- and no LLM in the routing path.

No network, no weights, no ChromaDB. Both tools are fakes so the WIRING is
what you are looking at, not the models.

The three things this demonstrates:

  1. ROUTING IS DETERMINISTIC. plan() is a pure function of the request.
     We pass NullLLM, which raises if called, to prove no model is consulted.
  2. TOOLS RUN IN PARALLEL and merge into ONE trace, with node order fixed
     by plan order rather than by who finished first.
  3. FAILURE IS SURVIVABLE. A tool that raises escalates instead of
     sinking the case.

Run:  python trace_03_orchestrator.py
"""

import time

from teta_agent.llm import EchoLLM, NullLLM
from teta_agent.orchestrator import CaseRequest, Orchestrator
from teta_agent.tools.adapters import GuidelinesTool, PathologyTool
from teta_agent.evidence import EvidenceLedger


# ── A fake retriever ───────────────────────────────────────────
# Stands in for the Chroma + MedEmbed + reranker stack, which takes ~30s to
# load. Sleeps briefly so the parallelism is observable.

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


def fake_search(query, n_results):
    """Return EvidenceRefs with stable P-labels, like the real pipeline."""
    time.sleep(0.25)  # pretend to embed + rerank
    ledger = fake_search.ledger
    refs = []
    for chunk_id, text, meta, score in PASSAGES[:n_results]:
        refs.append(ledger.add(chunk_id, text, meta, similarity=score))
    return refs


fake_search.ledger = EvidenceLedger()


def slow_pathology(slide_path, slide_id=None, top_k_tiles=5):
    """The stub, plus a delay so parallel vs serial is measurable."""
    from teta_agent.tools.pathology import classify_slide

    time.sleep(0.25)
    return classify_slide(slide_path, slide_id=slide_id,
                          force_subtype="B2", top_k_tiles=top_k_tiles)


def banner(text):
    print("\n" + "=" * 68)
    print(text)
    print("=" * 68)


def main():
    tools = [
        PathologyTool(classify_fn=slow_pathology),
        GuidelinesTool(search_fn=fake_search),
    ]

    # ── 1. Routing uses no model ───────────────────────────────
    banner("1. ROUTING IS MODEL-FREE (llm=NullLLM, which raises if called)")

    orch = Orchestrator(tools=tools, llm=NullLLM())
    request = CaseRequest(
        question="Is postoperative radiotherapy indicated for stage II B2 thymoma?",
        slide_path="/slides/UC-0042.svs",
        case_id="UC-0042",
    )

    print("\nplan() is a pure function of the request:")
    for tool in orch.plan(request):
        print("   selected:", tool.name)

    print("\nSlide only -> classifier alone:")
    for tool in orch.plan(CaseRequest(slide_path="/slides/x.svs")):
        print("   selected:", tool.name)

    print("\nQuestion only -> guidelines alone:")
    for tool in orch.plan(CaseRequest(question="What is a B3 thymoma?")):
        print("   selected:", tool.name)

    result = orch.run_case(request, synthesize=False)
    print("\nFull case with synthesize=False completed in {:.2f}s".format(
        result.wall_s))
    print("NullLLM was never called -> routing genuinely needs no model.")

    # ── 2. Parallel dispatch, one trace ────────────────────────
    banner("2. PARALLEL DISPATCH -> ONE MERGED TRACE")

    for tool_result in result.tool_results:
        print("\n  {} ({:.2f}s)".format(tool_result.tool_name,
                                        tool_result.elapsed_s))
        print("    {}".format(tool_result.summary))
        print("    {} uris".format(len(tool_result.uris)))

    serial = sum(r.elapsed_s for r in result.tool_results)
    print("\n  sum of tool times : {:.2f}s".format(serial))
    print("  wall clock        : {:.2f}s  <- ran concurrently".format(
        result.wall_s))

    print("\n  Merged trace nodes:")
    for node in result.trace.all_nodes():
        print("    {} [{}] {}".format(node.node_id, node.kind,
                                      node.summary[:60]))

    print("\n  ARIA payload for node t1:")
    payload = result.trace.get("t1").to_aria_payload(result.case_id)
    print("    evidenceRefs:", payload["evidenceRefs"][:2], "...")

    # ── 3. Escalation ──────────────────────────────────────────
    banner("3. ESCALATION AND FAILURE HANDLING")

    print("\nEscalations raised on this case:")
    for item in result.escalations:
        print("   !", item)
    print("\nneeds_human():", result.needs_human())

    def broken(slide_path, slide_id=None, top_k_tiles=5):
        raise RuntimeError("GPU node unreachable")

    degraded = Orchestrator(
        tools=[PathologyTool(classify_fn=broken),
               GuidelinesTool(search_fn=fake_search)],
        llm=EchoLLM(),
    )
    bad = degraded.run_case(request)
    print("\nWith the classifier failing:")
    print("   failed tools :", bad.failed_tools())
    print("   guidelines   :", bad.by_tool("guidelines_rag").ok)
    print("   escalations  :", len(bad.escalations))
    print("   -> the case still produced a result rather than crashing.")

    # ── 4. The swap ────────────────────────────────────────────
    banner("4. GOING LIVE: WHAT ACTUALLY CHANGES")
    print("""
Only the injected callables change -- no orchestrator edits:

    Orchestrator(
        tools=[PathologyTool(classify_fn=slideflow_hcord_predict),
               GuidelinesTool(search_fn=chroma_search)],
        llm=OpenAICompatLLM(model="argo:gpt-5"),
    )

CPU footprint of what you just ran: the orchestrator itself is pure Python
and does no matrix math. Retrieval embeddings already run device="cpu".
Only the synthesis model needs a GPU, and it is remote behind base_url.
""")


if __name__ == "__main__":
    main()
