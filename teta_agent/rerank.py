"""Cross-encoder reranking: score the query and passage *together*.

WHY THIS MODULE EXISTS
----------------------
The vector search is a **bi-encoder**: it embeds the question and every
chunk separately, then compares the vectors. That is fast enough to scan
337 chunks, but it never actually reads the question against the passage --
it only asks "are these two texts broadly about the same thing?"

A **cross-encoder** reads both at once and scores the pair directly. It is
far too slow to run over the whole corpus (it must do one forward pass per
candidate), which is exactly why this is a two-stage design:

    bi-encoder   337 chunks -> 40 candidates    (fast, approximate)
    cross-encoder 40 candidates -> scored       (slow, accurate)
    diversify    scored -> 5 passages           (policy)

We use ``ncbi/MedCPT-Cross-Encoder``, trained on 255M PubMed click pairs,
so "is this passage a good answer to this biomedical question?" is the
exact question it was built for.

A MEASURED CAVEAT -- RERANKING DOES NOT REPLACE BIBLIOGRAPHY DEMOTION
---------------------------------------------------------------------
Asked "indications for postoperative radiotherapy in stage II thymoma",
MedCPT scored a reference-list entry at **0.94**, against 1.00 for the
genuine recommendation::

    1.00  "Postoperative radiotherapy is recommended for stage II..."
    0.94  "51. Sugie C, Shibamoto Y. Invasive thymoma. J Thorac Oncol..."
    0.03  "Type B2 thymoma shows increased epithelial cells..."

The bibliography entry is keyword-dense and topically perfect; it simply
contains no assertion. The cross-encoder has no notion of "this is a
citation list, not prose". So ``retrieval.diversify`` still applies its
bibliography penalty *after* reranking. The two filters are orthogonal and
both are needed.

PORTABILITY CONTRACT (ARIAPlatform)
-----------------------------------
``Reranker`` takes ``(query, candidates)`` where candidates are
``(id, text, metadata, score)`` tuples and returns the same shape with
scores replaced. No thymoma knowledge, no ChromaDB. Swapping MedCPT for a
general-domain reranker is a constructor argument.
"""

from typing import List, Optional, Sequence, Tuple

#: Biomedical cross-encoder. Trained on PubMed search logs.
DEFAULT_RERANK_MODEL = "ncbi/MedCPT-Cross-Encoder"

#: Cross-encoder input budget. Query + passage must fit; longer passages are
#: truncated by the tokenizer, which is acceptable here because the opening
#: of a chunk carries most of its topical signal.
MAX_LENGTH = 512

#: MedCPT emits large unbounded relevance logits. ``CrossEncoder`` defaults
#: to a sigmoid for single-output models, which squashes anything above
#: ~+8 to 1.0 -- measured here, that tied the *median* candidate with the
#: best one and silently destroyed the ranking. Identity keeps the logits
#: ordered. Scores are therefore not probabilities and are only ever
#: compared against each other, never thresholded.
RERANK_ACTIVATION = "identity"


class Reranker:
    """Lazy-loading cross-encoder scorer.

    The model is loaded on first use rather than at import, so starting the
    chat does not pay a ~9 s model load before the user has asked anything.

    Args:
        model_name: HuggingFace id of a cross-encoder.
        device: Torch device string.
        max_length: Token budget for the (query, passage) pair.
    """

    def __init__(self, model_name=DEFAULT_RERANK_MODEL, device="cpu",
                 max_length=MAX_LENGTH):
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self._model = None
        self._failed = False

    @property
    def available(self):
        """Whether the model can be used (False once a load has failed)."""
        return not self._failed

    def _load(self):
        if self._model is not None or self._failed:
            return self._model
        try:
            import torch
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(
                self.model_name,
                device=self.device,
                max_length=self.max_length,
                activation_fn=torch.nn.Identity(),
            )
        except Exception as exc:  # pragma: no cover - environment dependent
            # Degrade to no reranking rather than killing the session. A
            # missing model should cost accuracy, not availability.
            print(f"   ⚠️  Reranker unavailable ({exc}); using vector order.")
            self._failed = True
        return self._model

    def score(self, query, passages):
        """Raw cross-encoder scores for one query against many passages.

        Returns:
            List of floats, same length and order as ``passages``. Empty
            list if the model could not be loaded.
        """
        if not passages:
            return []
        model = self._load()
        if model is None:
            return []
        pairs = [(query, text) for text in passages]
        raw = model.predict(pairs, show_progress_bar=False)
        return [float(x) for x in raw]

    def rerank(self, query, candidates, top_k=None):
        """Rescore and reorder candidates by cross-encoder relevance.

        Args:
            query: The user's question.
            candidates: ``[(chunk_id, text, metadata, vector_score), ...]``.
            top_k: Optionally truncate after reranking.

        Returns:
            ``(reranked, report)``. ``reranked`` has the same tuple shape
            with the **cross-encoder score** in the score slot, sorted best
            first. ``report`` records what changed, so the reordering is
            auditable rather than silent: ``reranked`` (bool),
            ``model``, ``promoted`` (count of candidates that moved into the
            top 5 from outside it), ``top_vector`` / ``top_cross`` ids.
        """
        if not candidates:
            return [], {"reranked": False, "reason": "no candidates"}

        texts = [text or "" for _, text, _, _ in candidates]
        scores = self.score(query, texts)

        if not scores:
            return list(candidates), {
                "reranked": False,
                "reason": "model unavailable",
                "model": self.model_name,
            }

        # Rank by the original vector score so we can measure movement.
        by_vector = sorted(
            range(len(candidates)),
            key=lambda i: candidates[i][3],
            reverse=True,
        )
        vector_top5 = set(by_vector[:5])

        order = sorted(
            range(len(candidates)), key=lambda i: scores[i], reverse=True
        )
        cross_top5 = set(order[:5])

        reranked = [
            (candidates[i][0], candidates[i][1], candidates[i][2], scores[i])
            for i in order
        ]
        if top_k is not None:
            reranked = reranked[:top_k]

        report = {
            "reranked": True,
            "model": self.model_name,
            "considered": len(candidates),
            "promoted": len(cross_top5 - vector_top5),
            "top_vector": candidates[by_vector[0]][0] if by_vector else None,
            "top_cross": candidates[order[0]][0] if order else None,
        }
        return reranked, report


#: Process-wide default, so the model is loaded at most once per session.
_DEFAULT: Optional[Reranker] = None


def get_reranker(model_name=DEFAULT_RERANK_MODEL, device="cpu"):
    """Shared ``Reranker`` instance, created on first call."""
    global _DEFAULT
    if _DEFAULT is None or _DEFAULT.model_name != model_name:
        _DEFAULT = Reranker(model_name=model_name, device=device)
    return _DEFAULT
