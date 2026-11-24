"""A/B test: does stating corpus scope change the synthesis answer?

The open question from the corpus-coverage work. Wiring the scope text into
the prompt was verified; whether the model *uses* it was not.

Design: identical retrieval, identical model, identical system prompt. The
only difference is whether ``corpus_scope`` is present in the tool detail.
Retrieval runs ONCE per question and both arms reuse those passages, so any
difference in the answers cannot come from retrieval variance.

Questions are chosen as a contrast:
  - immunotherapy in thymic carcinoma -> OUT of scope (treatment stops 2015)
  - WHO type B2 histological criteria -> IN scope (3 classification docs)

A useful result is asymmetric: scope should change the first answer and
leave the second alone. If it suppresses the second too, the text is
causing over-refusal and needs softening.

Run:  ./.venv/bin/python ab_corpus_scope.py
"""

import os
import sys

from teta_agent import corpus
from teta_agent.evidence import EvidenceLedger
from teta_agent.llm import OpenAICompatLLM
from teta_agent.orchestrator import SYNTHESIS_SYSTEM_PROMPT

#: Chroma collection name. Defined in build_db.py, which imports chromadb
#: and pypdf at module scope, so it is restated rather than imported.
COLLECTION_NAME = "teta_guidelines"

QUESTIONS = [
    ("OUT of scope", "What is the role of immunotherapy in thymic carcinoma?"),
    ("IN scope", "What are the histological criteria for WHO type B2 thymoma?"),
]


def load_retrieval():
    """Open the vector store and reranker from rag_chat's configuration."""
    import chromadb
    from chromadb.utils.embedding_functions import (
        SentenceTransformerEmbeddingFunction,
    )
    import rag_chat

    # The embedding model must match the store that built it -- vectors
    # from different models are not comparable.
    if rag_chat.VECTORSTORE_DIR.exists():
        store, embed_model = rag_chat.VECTORSTORE_DIR, rag_chat.EMBED_MODEL
    else:
        store = rag_chat.FALLBACK_VECTORSTORE_DIR
        embed_model = rag_chat.FALLBACK_EMBED_MODEL
    print("store: {} ({})".format(store, embed_model))

    embed_fn = SentenceTransformerEmbeddingFunction(
        model_name=embed_model, device="cpu",
    )
    client = chromadb.PersistentClient(path=str(store))
    collection = client.get_collection(
        COLLECTION_NAME, embedding_function=embed_fn,
    )
    print("passages: {}".format(collection.count()))

    reranker = None
    try:
        from teta_agent.rerank import get_reranker

        reranker = get_reranker()
    except Exception as exc:  # reranking is a bonus, not a precondition
        print("reranker unavailable ({}); using vector order".format(exc))
    return collection, reranker


def build_block(refs):
    """Render passages exactly as GuidelinesTool would."""
    return "\n\n".join(
        "[{}] {}\n{}".format(r.label, r.citation(), r.text) for r in refs
    )


def build_prompt(question, block, scope):
    """Mirror Orchestrator.build_prompt for a single guidelines result."""
    parts = [p for p in (scope, block) if p]
    return "## guidelines_rag\n{}\n\n# Question\n{}".format(
        "\n\n".join(parts), question
    )


def main():
    import rag_chat

    collection, reranker = load_retrieval()
    llm = OpenAICompatLLM(api_key=rag_chat.resolve_credential())
    print("model: {}\n".format(llm.model))

    for tag, question in QUESTIONS:
        print("=" * 68)
        print("{}  --  {}".format(tag, question))
        print("=" * 68)

        # Retrieve once; both arms see identical evidence.
        ledger = EvidenceLedger()
        hits, _ = rag_chat.retrieve(collection, question, 1, reranker)
        refs = rag_chat.record_hits(ledger, hits, 1)
        block = build_block(refs)
        print("retrieved {} passages from {}\n".format(
            len(refs), sorted({r.source for r in refs})))

        for arm, scope in (
            ("WITHOUT corpus scope", None),
            ("WITH corpus scope", corpus.CORPUS_SCOPE),
        ):
            prompt = build_prompt(question, block, scope)
            try:
                answer = llm.complete(SYNTHESIS_SYSTEM_PROMPT, prompt)
            except Exception as exc:
                print("--- {} ---\nFAILED: {}\n".format(arm, exc))
                continue
            print("--- {} ---".format(arm))
            print(answer.strip())
            print()


if __name__ == "__main__":
    sys.exit(main())
