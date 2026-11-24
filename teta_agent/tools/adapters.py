"""Tool adapters: the seam between the orchestrator and real capabilities.

WHY THIS MODULE EXISTS
----------------------
The orchestrator must not know what a thymoma is, and the classifier must not
know what an orchestrator is. Adapters hold the translation, and they are the
single place to edit when a stub becomes real.

EVERY ADAPTER SATISFIES ONE PROTOCOL::

    name                      stable str, used for trace node ids
    applies_to(request)       -> bool, is my input present?
    run(request)              -> ToolResult
    schema()                  -> OpenAI function schema
    invoke(request, args)     -> ToolResult, from LLM-supplied arguments

TWO ENTRY POINTS, ONE BODY
--------------------------
``run(request)`` is the deterministic path: inputs come off the
``CaseRequest``. ``invoke(request, arguments)`` is the agent path: inputs
come from the model. Both funnel into the same implementation, so the two
orchestrators cannot drift apart in what a tool actually does -- only in who
decided to call it and with what.

The split in *who supplies what* is deliberate and not symmetric:

    GuidelinesTool   query       <- the MODEL (this is the point)
    PathologyTool    slide_path  <- the ORCHESTRATOR, never the model

A model naming a file path is a hallucinated-path and path-traversal risk
with no upside; a model composing a search query is the entire benefit of
having an agent. See each ``schema()`` for the argument surface.

THE SWAP POINTS
---------------
Each adapter takes a *callable* rather than importing an implementation, so
replacing a stub never touches orchestrator code:

    PathologyTool(classify_fn=...)   default: the stub
    GuidelinesTool(search_fn=...)    required: no default, see below

Going live with Sacco weights::

    from slideflow.model import load            # real inference
    def real_classify(slide_path, slide_id=None, **kw):
        return load("hcord").predict(slide_path)   # -> PathologyPrediction
    PathologyTool(classify_fn=real_classify)

``GuidelinesTool`` has no default on purpose. The Chroma collection, the
embedding model, and the reranker all live in ``rag_chat.py`` and take ~30 s
to load; importing that at module scope would make even offline tests slow
and would bind this package to ChromaDB. Callers inject a search callable.
"""

from teta_agent import corpus
from teta_agent.orchestrator import ToolResult
from teta_agent.tools import pathology as pathology_stub


class ToolNotApplicable(ValueError):
    """A tool was called with arguments it cannot act on.

    Distinct from a tool *failure*: nothing broke, the request was simply
    invalid -- classify a case with no slide, search with an empty query.
    The agent loop turns this into a message back to the model so it can
    correct itself, rather than escalating to a human for what is a
    recoverable mistake.
    """


class PathologyTool:
    """Wraps the TET subtype classifier.

    Args:
        classify_fn: Callable ``(slide_path, slide_id=None, **kwargs)`` that
            returns an object shaped like ``PathologyPrediction``. Defaults
            to the stub.
        name: Trace-facing tool name.
        top_k_tiles: How many attention tiles to surface.
        review_threshold: Coarse-confidence floor before escalating.
    """

    def __init__(self, classify_fn=None, name="pathology_classifier",
                 top_k_tiles=5, review_threshold=0.70):
        self.classify_fn = classify_fn or pathology_stub.classify_slide
        self.name = name
        self.top_k_tiles = top_k_tiles
        self.review_threshold = review_threshold

    def applies_to(self, request):
        """Run only when a slide is attached."""
        return request.has_slide()

    def schema(self):
        """OpenAI function schema for LLM-driven tool calling.

        ``slide_path`` is deliberately NOT a model-supplied argument. The
        slide attached to the case is a fact about the case, and letting a
        model name a file path invites both hallucinated paths and a path
        traversal. The model may only decide *whether* to classify and how
        many tiles to see; the orchestrator supplies the path.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Classify the whole-slide image attached to this case "
                    "into a WHO thymic epithelial tumour subtype. Returns a "
                    "coarse group (As/Bs/TC) and a six-class subtype, each "
                    "with its own confidence, plus high-attention tiles. "
                    "Call this once when a slide is attached and the "
                    "histology is relevant."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reason": {
                            "type": "string",
                            "description": (
                                "Why this call is needed now. Recorded in "
                                "the audit trace."
                            ),
                        }
                    },
                    "required": [],
                },
            },
        }

    def invoke(self, request, arguments):
        """Run from LLM-supplied arguments.

        Args:
            request: The case, which supplies the slide path.
            arguments: Parsed model arguments. Only ``reason`` is read.

        Raises:
            ToolNotApplicable: If no slide is attached. The model can ask
                for a classification on a slideless case; this is where that
                fails closed.
        """
        if not request.has_slide():
            raise ToolNotApplicable(
                "no slide is attached to this case, so it cannot be classified"
            )
        result = self.run(request)
        reason = (arguments or {}).get("reason")
        if reason:
            result.detail["call_reason"] = str(reason)[:300]
        return result

    def run(self, request):
        prediction = self.classify_fn(
            request.slide_path,
            slide_id=request.case_id,
            top_k_tiles=self.top_k_tiles,
        )
        detail = prediction.to_trace_detail()

        # Translate the prediction's self-assessment into the orchestrator's
        # vocabulary. The tool owns the clinical judgement; the orchestrator
        # only reads the 'escalate' key.
        reasons = []
        if getattr(prediction, "is_stub", False):
            reasons.append("STUB model - not valid for clinical use")
        if prediction.coarse_confidence < self.review_threshold:
            reasons.append(
                "low coarse confidence {:.2f}".format(
                    prediction.coarse_confidence
                )
            )
        if not prediction.agrees_with_coarse():
            reasons.append(
                "subtype {} outside coarse group {}".format(
                    prediction.subtype, prediction.coarse_group
                )
            )
        if reasons:
            detail["escalate"] = "; ".join(reasons)

        detail["prompt_block"] = self._prompt_block(prediction)

        return ToolResult(
            tool_name=self.name,
            ok=True,
            summary=prediction.summary(),
            uris=prediction.uris(),
            detail=detail,
            payload=prediction,
        )

    def _prompt_block(self, prediction):
        """Render the prediction for the synthesis model.

        Both confidences are stated because the six-class call is materially
        less reliable than the coarse one (77.7% vs 91.1% in the paper), and
        a model shown only one number will treat them as equally solid.
        """
        lines = [
            "Slide: {}".format(prediction.slide_id),
            "Coarse group: {} (confidence {:.2f})".format(
                prediction.coarse_group, prediction.coarse_confidence
            ),
            "WHO six-class subtype: {} (confidence {:.2f})".format(
                prediction.subtype, prediction.subtype_confidence
            ),
            "Tiles analysed: {}".format(prediction.n_tiles),
        ]
        if getattr(prediction, "is_stub", False):
            lines.append(
                "WARNING: fabricated by a STUB model. State this in any "
                "answer; do not present it as a diagnosis."
            )
        return "\n".join(lines)


class GuidelinesTool:
    """Wraps retrieval over the guideline corpus.

    Args:
        search_fn: Callable ``(query, n_results)`` returning an object with
            ``.refs`` (``EvidenceRef`` list) and optionally
            ``.prompt_block()`` -- i.e. a ``RetrievalResult``. A plain list
            of refs is also accepted.
        ledger: Optional ``EvidenceLedger``, so labels stay stable across
            turns of the same case.
        n_results: Passages to retrieve after reranking.
    """

    def __init__(self, search_fn, ledger=None, name="guidelines_rag",
                 n_results=5):
        self.search_fn = search_fn
        self.ledger = ledger
        self.name = name
        self.n_results = n_results

    def applies_to(self, request):
        """Run only when there is a question to answer."""
        return request.has_question()

    def schema(self):
        """OpenAI function schema for LLM-driven tool calling.

        Unlike the classifier, ``query`` IS model-supplied, and that is the
        substantive difference between the two orchestrators. A deterministic
        run searches the clinician's words verbatim; an agent run searches
        whatever the model decided to ask -- typically a better query, since
        it can fold in the pathology result, but one no human vetted.

        The query is therefore recorded in the trace as a first-class field.
        ``record_search`` exists for exactly this: "was the right question
        asked?" stops being rhetorical once a model is asking.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Search the thymic-tumour clinical guideline corpus and "
                    "return the most relevant passages, each with a stable "
                    "[P] citation label. Call this before making any factual "
                    "clinical claim. You may call it more than once with "
                    "different queries to cover separate sub-questions."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": (
                                "The search query. Write it as a specific "
                                "clinical question, and fold in findings "
                                "already established this case (e.g. the "
                                "subtype and stage) rather than repeating "
                                "the user's wording verbatim."
                            ),
                        }
                    },
                    "required": ["query"],
                },
            },
        }

    def invoke(self, request, arguments):
        """Run from LLM-supplied arguments.

        Raises:
            ToolNotApplicable: If the model supplied no usable query. An
                empty query would still return five passages -- plausible
                ones, for nothing in particular -- so it fails closed.
        """
        query = str((arguments or {}).get("query", "")).strip()
        if not query:
            raise ToolNotApplicable(
                "a non-empty 'query' argument is required to search the "
                "guideline corpus"
            )
        return self._search(query, model_authored=True)

    def run(self, request):
        return self._search(request.question, model_authored=False)

    def _search(self, query, model_authored):
        found = self.search_fn(query, self.n_results)
        refs = getattr(found, "refs", found) or []

        if hasattr(found, "prompt_block"):
            block = found.prompt_block()
        else:
            block = "\n\n".join(
                "[{}] {}\n{}".format(r.label, r.citation(), r.text)
                for r in refs
            )

        detail = {
            "tool": self.name,
            "query": query,
            # Who wrote the query. A reviewer reading the trace must be able
            # to tell a clinician's words from a model's paraphrase without
            # cross-referencing anything.
            "query_author": "model" if model_authored else "user",
            "n_passages": str(len(refs)),
            "prompt_block": block,
            # What the model cannot infer from the passages themselves: the
            # shape of the library they came from. A question outside the
            # corpus still returns five plausible passages, so absence has
            # to be stated. See corpus.py for why this is not a score.
            "corpus_scope": corpus.CORPUS_SCOPE,
        }

        # Source concentration is a fact, not a judgement: five passages
        # from one document is one guideline's position, not consensus.
        sources = {r.source for r in refs if getattr(r, "source", None)}
        detail["source_count"] = str(len(sources))
        if len(sources) == 1 and len(refs) > 1:
            detail["caveat"] = (
                "All passages came from a single document ({}) - this "
                "reflects one guideline's position, not a consensus across "
                "sources.".format(next(iter(sources)))
            )

        if not refs:
            # Empty retrieval is not an error, but answering a clinical
            # question with zero grounding is exactly when a human is needed.
            detail["escalate"] = "no supporting passages retrieved"

        return ToolResult(
            tool_name=self.name,
            ok=True,
            summary="retrieved {} passages for: {}".format(
                len(refs), query[:80]
            ),
            uris=[r.uri for r in refs],
            detail=detail,
            payload=refs,
        )
