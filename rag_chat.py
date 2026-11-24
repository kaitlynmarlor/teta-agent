"""RAG Chat: Ask questions answered by your local guideline database.

The RAG loop, plus a reasoning trace:
  1. Your question -> ChromaDB vector search -> 40 candidate chunks
  2. A biomedical CROSS-ENCODER reranks those 40 by true relevance
  3. Bibliography demotion + per-document cap -> 5 passages, stable [P] labels
  4. Labelled passages injected into the prompt -> grounded answer
  5. The answer is parsed back into claims, each linked to the passages it
     cites, producing a filtered trace of what the answer actually rests on

Step 5 is what makes the output auditable. Steps 1-4 can only show you what
was *retrieved*; the trace shows what was *used*, which is a strictly
smaller and more interesting set.

WHY TWO SCORING STAGES: the vector search embeds question and chunk
separately, which is fast but approximate. The cross-encoder reads both
together and is far more accurate, but too slow for the whole corpus. So
the first stage narrows 337 -> 40 and the second stage ranks those properly.

Commands inside the session:
  summary   whole case on one screen — exceptions first   [triage]
  trace     claim-by-claim support for the last answer    [investigate]
  raw       the unfiltered graph + what got dropped       [debug]
  evidence  list every passage cited so far, with URIs
  open P1   open that passage in the real PDF, highlighted [verify]
  no-rag    toggle retrieval off, to compare against an ungrounded answer
  quit

Run:  python rag_chat.py
"""

import os
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import openai

from teta_agent.claims import (
    populate_trace,
    render_case_summary,
    render_clinician_report,
)
from teta_agent.evidence import EvidenceLedger
from teta_agent.rerank import get_reranker
from teta_agent.retrieval import diversify
from teta_agent.trace import ReasoningTrace
from teta_agent.viewer import CitationViewer

# ── Configuration ──────────────────────────────────────────────

#: Biomedical index built by ``build_db.py``. Falls back to the original
#: general-domain store if this has not been built yet.
VECTORSTORE_DIR = Path("vectorstore_v2")
FALLBACK_VECTORSTORE_DIR = Path("vectorstore_data")

#: Must match whatever ``build_db.py`` embedded with. A mismatch here does
#: not error -- it silently returns nonsense, because the query vector and
#: the stored vectors come from different spaces.
EMBED_MODEL = "abhinand/MedEmbed-large-v0.1"
FALLBACK_EMBED_MODEL = "BAAI/bge-large-en-v1.5"

#: Source PDFs, for the clickable citation viewer.
GUIDELINES_DIR = Path("data/guidelines")
VIEWER_PORT = 8731

VLLM_BASE_URL = "http://localhost:44497/v1"

# Left unset on purpose -- see resolve_credential(). argo-proxy treats this
# bearer token as your ARGO *username*, not as an API key, so a wrong value
# fails with 403 "the username is not registered in ARGO" rather than the
# 401 you would expect from a bad key.
VLLM_API_KEY = ""

MODEL = "argo:claude-4.6-opus"
N_RESULTS = 5

# Pull a wide candidate pool, then re-rank it down to N_RESULTS. Plain
# top-5 returned 19 consecutive ESMO chunks on a WHO-histology question,
# with a score spread of only 0.048 — statistically tied, so the ordering
# among them was close to arbitrary. See teta_agent/retrieval.py.
CANDIDATE_POOL = 40
MAX_PER_SOURCE = 2

#: Rerank the candidate pool with a biomedical cross-encoder before
#: applying diversity policy. Set False to compare against vector-only
#: ordering -- useful for showing what reranking actually buys.
USE_RERANK = True

# ── The prompt template ────────────────────────────────────────
# This is where "Augmentation" happens — we inject retrieved
# context into the prompt so the LLM can reason over it.

SYSTEM_PROMPT = """You are a clinical decision support assistant
specializing in thymic epithelial tumors. Answer questions using
ONLY the provided context from clinical guidelines and literature.

CITATION RULES — these are strict, and the two kinds are different:

1. RETRIEVED PASSAGES are labelled [P1], [P2], ... ("P" for passage). Each
   label is a stable address for one specific passage. Cite the label
   exactly as given, e.g.
   "Postoperative radiotherapy may be considered [P1]".
   - Never invent a label that does not appear in the context.
   - Never renumber them. [P4] stays [P4] even if it appears last.

2. STUDIES NAMED INSIDE A PASSAGE are NOT things we retrieved. If [P3]
   says "Loehrer et al. reported a 57% response rate", you must write:
       "Loehrer et al. reported a 57% response rate (as cited in [P3])"
   Do NOT write "[P3] reported 57%" — P3 is the review article, not the
   Loehrer study. We do not hold the Loehrer paper. This distinction tells
   the clinician which claims are first-hand and which are second-hand.

3. ESMO EVIDENCE GRADING is a THIRD, separate thing. Passages often contain
   notation like "[II, B]", "[III, C]" or "[IV, E]" — a Roman numeral level
   of evidence plus a letter grade of recommendation. Reproduce that
   notation verbatim when the passage carries it, and never confuse it with
   a passage label. In particular ESMO grade **E** means "strong evidence
   against efficacy — never recommended"; it is not a reference pointer.

Evidence rules:
- Base your answer strictly on the provided context
- You may ONLY state numbers, percentages, response rates, and
  statistics that appear VERBATIM in the provided context. If a
  study is referenced but specific data is not provided, say
  "specific results not available in the provided context"
- Do NOT supplement the context with your own knowledge
- If the context doesn't contain enough information to answer,
  say so explicitly — do NOT make up information
- Be concise but thorough
"""

def resolve_credential():
    """Get the ARGO username, from env or by asking at startup.

    Precedence: ``VLLM_API_KEY`` constant -> ``ARGO_USER`` env var -> prompt.

    Asked for at startup rather than at first generation so a typo surfaces
    before you have spent 30 seconds loading the embedding model and typing
    a question.

    Note this is a *username*, not a secret, so it is echoed normally. If
    argo-proxy ever moves to real API keys this should switch to
    ``getpass.getpass`` to keep it off the screen and out of shell history.
    """
    if VLLM_API_KEY:
        return VLLM_API_KEY

    from_env = os.environ.get("ARGO_USER", "").strip()
    if from_env:
        print(f"Using ARGO_USER from environment: {from_env}")
        return from_env

    print("\nargo-proxy authenticates with your ARGO *username*, not an")
    print("API key. (Set ARGO_USER in your shell to skip this prompt.)")
    value = input("ARGO username: ").strip()
    while not value:
        value = input("Required — ARGO username: ").strip()
    return value


def retrieve(collection, question, turn, reranker=None):
    """Search, rerank, then apply diversity and bibliography policy.

    Three stages, each doing a job the others cannot:

    1. **Vector search** narrows 337 chunks to 40. Fast, approximate.
    2. **Cross-encoder rerank** scores those 40 by reading question and
       passage together. Accurate, too slow for the full corpus.
    3. **Diversify** applies policy: demote bibliography, cap per document.

    Stage 3 still runs after reranking, and deliberately so. Measured on
    this corpus, the cross-encoder scored a reference-list entry at 0.94
    against 1.00 for the genuine recommendation -- it has no concept of
    "this is a citation list, not prose". The two filters are orthogonal.

    Returns ``(hits, report)`` where ``hits`` is
    ``[(chunk_id, document, metadata, score), ...]``. With reranking on,
    ``score`` is cross-encoder relevance rather than cosine similarity.
    """
    raw = collection.query(query_texts=[question], n_results=CANDIDATE_POOL)
    candidates = [
        (chunk_id, doc, meta, 1 - dist)
        for chunk_id, doc, meta, dist in zip(
            raw["ids"][0],
            raw["documents"][0],
            raw["metadatas"][0],
            raw["distances"][0],
        )
    ]

    rerank_report = {"reranked": False}
    if reranker is not None:
        candidates, rerank_report = reranker.rerank(question, candidates)

    hits, report = diversify(
        candidates, N_RESULTS, max_per_group=MAX_PER_SOURCE
    )
    report["rerank"] = rerank_report
    return hits, report


def record_hits(ledger, hits, turn):
    """Register this search's hits in the ledger and return their refs.

    The ledger — not the loop counter — decides each label. A chunk seen in
    an earlier turn keeps the label it was first given, so [P1] means the
    same passage for the whole conversation.
    """
    return [
        ledger.add(
            chunk_id,
            doc,
            meta,
            similarity=sim,
            first_seen_in=f"search_{turn}",
        )
        for chunk_id, doc, meta, sim in hits
    ]


def build_context_prompt(question, refs):
    """Build the augmented prompt from ledger-labelled passages."""
    context_block = "\n\n---\n\n".join(
        f"{ref.header()}\n{ref.text}" for ref in refs
    )

    return f"""Context from clinical guidelines:

{context_block}

---

Question: {question}

Provide a thorough answer based on the context above. Cite retrieved
passages by their exact [P] label. For any study named *inside* a
passage, attribute it to that study and add "(as cited in [P...])"."""


# ── Main loop ──────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("TETA RAG Chat")
    print(f"Model: {MODEL} (via Argo Proxy)")
    print("=" * 60)

    # Ask before the slow work, so a typo costs seconds not minutes.
    api_key = resolve_credential()

    # Prefer the biomedical index; fall back to the original store so the
    # chat still runs if build_db.py has not been re-run yet. The embedding
    # model MUST match the one the store was built with -- mismatched
    # vectors do not raise, they just return quiet nonsense.
    if VECTORSTORE_DIR.exists():
        store_dir, embed_model = VECTORSTORE_DIR, EMBED_MODEL
    else:
        store_dir, embed_model = FALLBACK_VECTORSTORE_DIR, FALLBACK_EMBED_MODEL
        print(f"\n⚠️  {VECTORSTORE_DIR}/ not found — using {store_dir}/")
        print("   Run build_db.py for biomedical embeddings.")

    client = chromadb.PersistentClient(path=str(store_dir))

    print(f"\nLoading embedding model: {embed_model}")
    embed_fn = SentenceTransformerEmbeddingFunction(
        model_name=embed_model,
        device="cpu",
    )

    collection = client.get_collection(
        name="teta_guidelines",
        embedding_function=embed_fn,
    )
    print(f"Connected to {store_dir}/ ({collection.count()} chunks)")

    # Lazy load: the cross-encoder is fetched on the first question, not
    # now, so startup does not pay for a feature that may go unused.
    reranker = get_reranker() if USE_RERANK else None
    if reranker is not None:
        print(f"Reranker: {reranker.model_name} (loads on first question)")

    print("\nCommands: summary | trace | raw | evidence | open P1 | "
          "no-rag | quit\n")

    # One client for the session rather than one per question -- it holds a
    # connection pool, and rebuilding it each turn was wasteful.
    llm = openai.OpenAI(base_url=VLLM_BASE_URL, api_key=api_key)

    use_rag = True

    # One ledger AND one trace for the whole session. Both are per-case, not
    # per-question: [P1] means the same passage across every question, and
    # the graph accumulates so a later answer can rest on a passage pulled
    # for an earlier one. That cross-turn reuse is real reasoning history,
    # and it only survives if these live outside the loop.
    ledger = EvidenceLedger()
    trace = ReasoningTrace(run_id="teta-session")
    turn = 0

    # Serves cited passages as highlighted PDF pages. Nothing is bound
    # until the first `open` command, so there is no background port in a
    # session that never uses it.
    viewer = CitationViewer(ledger, GUIDELINES_DIR, port=VIEWER_PORT)

    # Every answered question, in order. `last` is just answers[-1]; the
    # full list is what the case summary folds over.
    answers = []
    last = None

    while True:
        question = input("Question: ").strip()
        if not question:
            continue
        if question.lower() == "quit":
            break
        if question.lower() == "no-rag":
            use_rag = not use_rag
            mode = "ON ✅" if use_rag else "OFF ❌ (LLM only)"
            print(f"\n   RAG is now {mode}\n")
            continue
        if question.lower() == "evidence":
            if not len(ledger):
                print("\n   No evidence retrieved yet.\n")
            else:
                print(f"\n   Evidence ledger ({len(ledger)} passages):")
                for ref in ledger.all():
                    lines = ref.line_range()
                    suffix = f"  [{lines}]" if lines else ""
                    print(f"   [{ref.label}] {ref.citation()}{suffix}")
                    print(f"        {ref.uri}")
                print("\n   Tip: 'open P1' shows that passage in the PDF.\n")
            continue
        if question.lower().startswith("open"):
            # `open P3` — show the cited passage on its real page, with the
            # text highlighted. This is the last link in the audit chain:
            # claim -> passage -> the actual printed page.
            parts = question.split()
            if len(parts) < 2:
                print("\n   Usage: open P1\n")
                continue
            label = parts[1].strip("[]").upper()
            ok, message = viewer.open_label(label)
            if ok:
                print(f"\n   🔎 Opened {label} in your browser: {message}\n")
            else:
                print(f"\n   {message}\n")
            continue
        if question.lower() == "trace":
            if last is None:
                print("\n   No answer to trace yet. Ask a question first.\n")
            else:
                print()
                print(render_clinician_report(
                    trace,
                    last["answer_id"],
                    ledger,
                    second_hand_ids=last["second_hand"],
                    only_turn=last["turn"],
                ))
                print()
            continue
        if question.lower() == "summary":
            if not answers:
                print("\n   Nothing to summarise yet. Ask a question first.\n")
            else:
                print()
                print(render_case_summary(trace, answers, ledger))
                print()
            continue
        if question.lower() == "raw":
            # The teaching view: show the filter actually discarding things.
            if last is None:
                print("\n   Nothing recorded yet. Ask a question first.\n")
                continue
            kept = trace.filtered_to(last["answer_id"])
            kept_ids = {n.node_id for n in kept}
            this_turn = [n for n in trace.all_nodes()
                         if n.turn == last["turn"]]
            print(f"\n   Turn {last['turn']} recorded {len(this_turn)} nodes "
                  f"({len(trace)} in session):\n")
            for node in this_turn:
                mark = "✓" if node.node_id in kept_ids else "✗"
                parents = f" ← {node.parent_ids}" if node.parent_ids else ""
                print(f"   {mark} {node.node_id:7s} {node.kind:10s} "
                      f"{node.summary[:40]:40s}{parents}")
            print("\n   ✓ = reachable backwards from the answer (kept)")
            print("   ✗ = retrieved but nothing in the answer cites it\n")
            continue

        if use_rag:
            # Step 1: RETRIEVE
            turn += 1
            print(f"\n🔍 Retrieving relevant context...")
            hits, report = retrieve(collection, question, turn, reranker)

            # Step 1b: give every hit a conversation-stable label
            seen_before = set(ledger.known_labels())
            refs = record_hits(ledger, hits, turn)

            # Step 1c: record the search and its passages as graph nodes.
            # The tool id carries the turn number ("t3") because node ids
            # must be unique across the whole session, not just this loop
            # iteration. Evidence nodes reuse the ledger label as their id,
            # so a passage re-retrieved on a later turn maps onto the node
            # that already exists rather than duplicating it.
            tool_id = f"t{turn}"
            trace.record_search(tool_id, question, turn,
                                n_candidates=report.get("considered", 0),
                                report=report)
            for ref in refs:
                trace.record_evidence(ref, tool_id, turn)

            for ref in refs:
                flag = "" if ref.label not in seen_before else "  ♻️ seen earlier"
                print(f"   [{ref.label}] {ref.citation()} "
                      f"(similarity: {ref.similarity:.3f}){flag}")

            # Say out loud what retrieval chose NOT to show the model.
            print(f"   └─ {report['considered']} candidates → "
                  f"{len(report['groups'])} sources, "
                  f"{report['demoted']} reference-list chunks demoted")

            # Step 2: AUGMENT
            prompt = build_context_prompt(question, refs)
        else:
            # No RAG — just send the question directly
            prompt = question

        # Step 3: GENERATE
        print(f"\n💬 Generating answer ({'with RAG' if use_rag else 'LLM only'})...\n")

        try:
            response = llm.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
        except openai.AuthenticationError as exc:
            # Translate argo-proxy's 403 into the actual cause. Without this
            # the raw error says "authentication", which sends you looking
            # for an API key that does not exist.
            print(f"   ❌ ARGO rejected the username {api_key!r}.")
            print(f"      {exc}")
            print("      Restart and enter your ANL username.\n")
            continue
        except openai.APIConnectionError:
            print(f"   ❌ No answer from {VLLM_BASE_URL}.")
            print("      The port is an SSH tunnel — check it is still up.\n")
            continue

        answer = response.choices[0].message.content
        print(answer)

        # Step 4: TRACE — parse the answer back into claims.
        # Only meaningful with RAG on; an ungrounded answer has no labels to
        # link to, so there is nothing to trace and we say so rather than
        # rendering an empty graph.
        if use_rag:
            answer_id = f"answer{turn}"
            answer_id, claims, hallucinated, second_hand = populate_trace(
                trace, answer, ledger, turn, answer_node_id=answer_id
            )
            last = {"answer_id": answer_id, "turn": turn,
                    "question": question, "second_hand": second_hand}
            answers.append(last)

            kept = trace.filtered_to(answer_id)
            used = {n.node_id for n in kept}
            unused = [r.label for r in refs if r.label not in used]

            print("\n" + "-" * 60)
            print(f"📊 {len(claims)} cited claims   "
                  f"{len(refs) - len(unused)}/{len(refs)} passages used"
                  + (f"   unused: {', '.join(unused)}" if unused else ""))
            if hallucinated:
                # A label the model invented. Loud, because a fabricated
                # citation is worse than an uncited claim: it looks sourced.
                print(f"   ⚠️  CITED BUT NOT RETRIEVED: {hallucinated}")
            if not claims:
                print("   ⚠️  No sentence cited any passage — the answer is "
                      "ungrounded.")
            print("   'trace' = this answer in detail · "
                  "'summary' = whole case · 'raw' = graph")

        print("\n" + "-" * 60 + "\n")


if __name__ == "__main__":
    main()