"""STUB: TET histological subtype classifier.

WHAT THIS STANDS IN FOR
-----------------------
Sacco et al., "Deep learning discriminates thymic epithelial tumors'
histological subtypes using digital pathology" (*Annals of Oncology* 37(4),
2026). The real pipeline is:

    WSI -> tessellate into 224x224 tiles
        -> H-optimus-0 ViT  (1536-dim feature vector per tile)
        -> attention-based multiple-instance learning (Ilse et al.)
        -> MLP -> 7 logits
        -> hierarchical classifier -> subtype

This module reproduces that **interface** exactly -- same inputs, same
outputs, same 7-logit hierarchy -- and fabricates the numbers. No pixels are
read. It exists so the orchestrator, the reasoning trace, and the UI can be
built and tested before model weights are available.

THE 7-LOGIT HIERARCHY (paper, "Loss function design")
-----------------------------------------------------
The output is a seven-dimensional logit vector in three partitions::

    logits[0:3]  coarse group    As | Bs | TC
    logits[3:5]  A-subtype head  A  | AB         (read only when coarse == As)
    logits[5:7]  B-subtype head  2-bit ordinal   (read only when coarse == Bs)

The B head is an *ordinal* 2-bit code, not a 3-way softmax::

    B1 = [0, 0]      B2 = [0, 1]      B3 = [1, 1]

Successive classes differ by exactly one bit, encoding the
lymphocyte-predominant -> epithelial-predominant continuum. ``[1, 0]`` is
unreachable by construction; if it occurs we treat it as B2, the midpoint.

WHY THE HIERARCHY MATTERS CLINICALLY
------------------------------------
The coarse group drives treatment. The paper reports 91.1% accuracy
(kappa = 0.859) on the three-group scheme but only 77.7% (kappa = 0.716) on
six-class -- and notes that 60% of six-class errors stay *inside* the correct
coarse group, so they do not change management. A consumer of this tool
should therefore trust ``coarse_group`` more than ``subtype``, and the result
object keeps the two confidences separate so that asymmetry stays visible
rather than being collapsed into one number.

PORTABILITY
-----------
Pure stdlib. No torch, no numpy, no slide I/O. Deterministic given a seed.
"""

from dataclasses import dataclass, field
from typing import Dict, List
import hashlib
import math
import os

# ── Label vocabulary ────────────────────────────────────────────

COARSE_AS = "As"
COARSE_BS = "Bs"
COARSE_TC = "TC"
COARSE_GROUPS = (COARSE_AS, COARSE_BS, COARSE_TC)

#: Human-facing expansion of each coarse group.
COARSE_DESCRIPTIONS = {
    COARSE_AS: "Thymoma A / AB",
    COARSE_BS: "Thymoma B1 / B2 / B3",
    COARSE_TC: "Thymic carcinoma",
}

SUBTYPES = ("A", "AB", "B1", "B2", "B3", "TC")

#: Which coarse group each six-class subtype belongs to.
SUBTYPE_TO_COARSE = {
    "A": COARSE_AS,
    "AB": COARSE_AS,
    "B1": COARSE_BS,
    "B2": COARSE_BS,
    "B3": COARSE_BS,
    "TC": COARSE_TC,
}

#: 2-bit ordinal code for the B subtypes (paper, Figure 1C).
B_ORDINAL_CODE = {
    (0, 0): "B1",
    (0, 1): "B2",
    (1, 1): "B3",
    # Not reachable in a well-formed ordinal code; midpoint is the safe read.
    (1, 0): "B2",
}

#: Real tile geometry from the paper, surfaced so callers need not hardcode it.
TILE_PX = 224
FEATURE_DIM = 1536
FOUNDATION_MODEL = "H-optimus-0"

#: Per-coarse-group accuracy reported on the UChicago external validation set
#: (n=112). Used as the stub's *reliability* signal, not invented.
REPORTED_ACCURACY = {
    "three_group": 0.911,
    "six_class": 0.777,
    "tc_detection": 0.946,
}
REPORTED_KAPPA = {"three_group": 0.859, "six_class": 0.716}

#: Scheme for addressing a slide region, mirroring ``evidence.evidence_uri``.
#: A tile is evidence *of the same kind* as a retrieved passage -- it is a
#: locatable thing the model looked at -- so it gets a URI too.
SLIDE_SCHEME = "slide"


def tile_uri(slide_id, tile_index, scheme=SLIDE_SCHEME):
    """Canonical address for one tile of one slide.

    Mirrors the shape of ``teta_agent.evidence.evidence_uri`` so the trace can
    hold pathology tiles and guideline passages in the same
    ``evidenceRefs`` array without special-casing either.
    """
    return "{}://{}/tiles/{:06d}".format(scheme, slide_id, tile_index)


@dataclass(frozen=True)
class TileAttribution:
    """One high-attention tile: the pathology analogue of a cited passage.

    The real model produces two heatmaps -- attention (which tiles mattered)
    and per-subtype score (what each tile looked like). A single tile's row in
    those maps is exactly this record.

    Attributes:
        tile_index: Position in the tessellation, 0-based.
        uri: Addressable handle, e.g. ``slide://S-001/tiles/000042``.
        x: Tile's left edge in slide pixel coordinates.
        y: Tile's top edge in slide pixel coordinates.
        attention: Attention weight in [0, 1]; higher = more influential.
        subtype_scores: Per-subtype probability for THIS tile. Mirrors the
            subtype-score heatmap, so a caller can show "this region looks
            like AB even though the slide was called A".
    """

    tile_index: int
    uri: str
    x: int
    y: int
    attention: float
    subtype_scores: Dict[str, float] = field(default_factory=dict)

    def top_subtype(self):
        """Subtype this individual tile most resembles."""
        if not self.subtype_scores:
            return ""
        return max(self.subtype_scores.items(), key=lambda kv: kv[1])[0]

    def citation(self):
        """Human-facing one-liner, parallel to ``EvidenceRef.citation()``."""
        return "tile #{} at ({}, {}) - attention {:.2f}, looks like {}".format(
            self.tile_index, self.x, self.y, self.attention, self.top_subtype()
        )


@dataclass(frozen=True)
class PathologyPrediction:
    """What the classifier returns for one whole-slide image.

    Attributes:
        slide_id: Caller-supplied identifier for the slide.
        logits: The raw 7-vector, in the paper's partition order. Exposed
            because the hierarchy is only auditable if you can see it.
        coarse_group: ``As`` | ``Bs`` | ``TC`` -- the treatment-relevant call.
        coarse_confidence: Softmax probability of ``coarse_group``.
        coarse_probs: Full distribution over the three coarse groups.
        subtype: Six-class WHO call, e.g. ``"B2"``.
        subtype_confidence: Confidence in the six-class call. Always <= the
            coarse confidence, because the subtype decision is conditional on
            the coarse one being right.
        n_tiles: Tiles that survived tessellation -- the "bag" size.
        attributions: High-attention tiles, most influential first.
        model_version: Identifies the stub so results can never be silently
            mistaken for real inference.
        is_stub: Always True here. Consumers should refuse to put a stubbed
            prediction in front of a clinician.
    """

    slide_id: str
    logits: List[float]
    coarse_group: str
    coarse_confidence: float
    coarse_probs: Dict[str, float]
    subtype: str
    subtype_confidence: float
    n_tiles: int
    attributions: List[TileAttribution] = field(default_factory=list)
    model_version: str = "tet-hcord-stub-0.1"
    is_stub: bool = True

    # ── Derived views ──────────────────────────────────────────

    def is_carcinoma(self):
        """True when the slide is called thymic carcinoma.

        Worth its own method: TC is the branch that changes management most
        sharply, and the paper reports 100% sensitivity for it.
        """
        return self.coarse_group == COARSE_TC

    def agrees_with_coarse(self):
        """Whether the six-class call sits inside the coarse group.

        The two heads are decoded from different logit partitions, so they
        can in principle disagree. The paper's key safety finding -- 60% of
        six-class errors stay within the correct management group -- is only
        checkable if this is explicit.
        """
        return SUBTYPE_TO_COARSE.get(self.subtype) == self.coarse_group

    def needs_review(self, threshold=0.70):
        """Flag low-confidence or internally inconsistent predictions.

        This is the hook an orchestrator uses to decide whether to escalate
        to a human pathologist rather than proceeding automatically.
        """
        return (
            self.coarse_confidence < threshold
            or not self.agrees_with_coarse()
            or self.is_stub
        )

    def summary(self):
        """One-line human-facing result."""
        return (
            "{}: {} ({}), coarse {} p={:.2f} / six-class {} p={:.2f}".format(
                self.slide_id,
                self.subtype,
                COARSE_DESCRIPTIONS.get(self.coarse_group, self.coarse_group),
                self.coarse_group,
                self.coarse_confidence,
                self.subtype,
                self.subtype_confidence,
            )
        )

    def uris(self):
        """Tile URIs, ready to drop into an ARIA ``evidenceRefs`` array."""
        return [a.uri for a in self.attributions]

    def to_trace_detail(self):
        """Render as the ``detail`` dict of a ``TraceNode``.

        Values are stringified because ``TraceNode.detail`` is typed
        ``Dict[str, str]`` -- keeping that contract means the orchestrator can
        record a pathology call with no tool-specific code.
        """
        return {
            "tool": "pathology_classifier",
            "model": self.model_version,
            "slide_id": self.slide_id,
            "coarse_group": self.coarse_group,
            "coarse_confidence": "{:.3f}".format(self.coarse_confidence),
            "subtype": self.subtype,
            "subtype_confidence": "{:.3f}".format(self.subtype_confidence),
            "n_tiles": str(self.n_tiles),
            "needs_review": str(self.needs_review()),
            "is_stub": str(self.is_stub),
        }


# ── Hierarchy decoding (real logic, not stubbed) ────────────────
#
# Everything below this line is the paper's actual decoding rule. Only the
# *logits* are fabricated. Swapping in real weights means replacing
# ``_fake_logits`` and nothing else -- which is the point of the exercise.


def _softmax(values):
    """Numerically stable softmax over a short list."""
    if not values:
        return []
    top = max(values)
    exps = [math.exp(v - top) for v in values]
    total = sum(exps)
    return [e / total for e in exps]


def decode_logits(logits):
    """Apply the hierarchical classifier to a 7-logit vector.

    This is the paper's Figure 1C rule, implemented literally:
    coarse group from the first three logits; then *conditionally* read the
    A-head or the ordinal B-head.

    Args:
        logits: Seven floats. Partitioned ``[0:3]`` coarse, ``[3:5]`` A/AB,
            ``[5:7]`` B ordinal bits.

    Returns:
        Tuple of ``(coarse_group, coarse_probs, subtype, subtype_confidence)``.

    Raises:
        ValueError: If ``logits`` is not length 7. Failing loudly matters --
            a silently truncated vector would decode to a confident wrong
            subtype.
    """
    if len(logits) != 7:
        raise ValueError(
            "expected 7 logits (3 coarse + 2 A-head + 2 B-head), got {}".format(
                len(logits)
            )
        )

    coarse_probs_list = _softmax(list(logits[0:3]))
    coarse_probs = dict(zip(COARSE_GROUPS, coarse_probs_list))
    best_idx = max(range(3), key=lambda i: coarse_probs_list[i])
    coarse_group = COARSE_GROUPS[best_idx]
    coarse_conf = coarse_probs_list[best_idx]

    if coarse_group == COARSE_TC:
        # TC is a leaf: the coarse call *is* the subtype, so the two
        # confidences coincide.
        return coarse_group, coarse_probs, "TC", coarse_conf

    if coarse_group == COARSE_AS:
        a_probs = _softmax(list(logits[3:5]))
        subtype = "A" if a_probs[0] >= a_probs[1] else "AB"
        head_conf = max(a_probs)
    else:
        # Ordinal 2-bit head: each bit is an INDEPENDENT sigmoid, not a
        # softmax over two classes. Thresholding at 0.5 recovers the code.
        bits = tuple(1 if _sigmoid(v) >= 0.5 else 0 for v in logits[5:7])
        subtype = B_ORDINAL_CODE[bits]
        bit_confs = [max(p, 1.0 - p) for p in (_sigmoid(v) for v in logits[5:7])]
        head_conf = bit_confs[0] * bit_confs[1]

    # The subtype call is conditional on the coarse call, so the
    # probabilities multiply. This is why subtype_confidence is always the
    # smaller number -- and why six-class accuracy is lower than three-group.
    return coarse_group, coarse_probs, subtype, coarse_conf * head_conf


def _sigmoid(value):
    """Logistic function, overflow-safe for large-magnitude inputs."""
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_v = math.exp(value)
    return exp_v / (1.0 + exp_v)


# ── Fabricated inference ────────────────────────────────────────


def _rng_stream(seed_text):
    """Deterministic float stream in [0, 1) derived from a string.

    Uses SHA-256 rather than ``random.seed`` so results are stable across
    Python versions and unaffected by any other code touching global RNG
    state. Same slide id always yields the same prediction, which makes
    demos reproducible and tests non-flaky.
    """
    counter = 0
    while True:
        digest = hashlib.sha256(
            "{}:{}".format(seed_text, counter).encode("utf-8")
        ).digest()
        for i in range(0, 32, 4):
            chunk = digest[i : i + 4]
            yield int.from_bytes(chunk, "big") / 4294967296.0
        counter += 1


def _fake_logits(stream, force_subtype=None):
    """Invent a 7-logit vector.

    When ``force_subtype`` is given we generate logits that *decode to* it,
    rather than bolting the label on afterwards. That keeps the stub honest:
    the output always passes through the same decoder the real model's output
    would, so the decoder is exercised on every call.
    """
    # Baseline: mild noise everywhere.
    logits = [(next(stream) - 0.5) * 1.5 for _ in range(7)]

    if force_subtype is None:
        return logits

    if force_subtype not in SUBTYPES:
        raise ValueError(
            "unknown subtype {!r}; expected one of {}".format(
                force_subtype, ", ".join(SUBTYPES)
            )
        )

    coarse = SUBTYPE_TO_COARSE[force_subtype]
    # Push the intended coarse logit clear of the others. Magnitude varies so
    # confidences are not all identical.
    margin = 1.8 + next(stream) * 1.6
    logits[COARSE_GROUPS.index(coarse)] += margin

    if coarse == COARSE_AS:
        gap = 1.2 + next(stream) * 1.5
        if force_subtype == "A":
            logits[3] += gap
        else:
            logits[4] += gap
    elif coarse == COARSE_BS:
        # Drive each ordinal bit to the right side of 0.5.
        bits = {"B1": (0, 0), "B2": (0, 1), "B3": (1, 1)}[force_subtype]
        for offset, bit in enumerate(bits):
            strength = 1.5 + next(stream) * 1.8
            logits[5 + offset] = strength if bit else -strength

    return logits


def _fake_attributions(stream, slide_id, n_tiles, subtype, top_k):
    """Invent high-attention tiles consistent with the slide-level call.

    Attention weights decay across the ranking so the list looks like a real
    sorted attention head. Most tiles agree with the slide-level subtype; a
    minority dissent, mirroring Figure 2D, where a slide genuinely contains
    more than one subtype.
    """
    out = []
    k = max(0, min(top_k, n_tiles))
    # Grid geometry for plausible coordinates.
    cols = max(1, int(math.sqrt(n_tiles)))
    attention = 0.85 + next(stream) * 0.1

    for rank in range(k):
        tile_index = int(next(stream) * n_tiles) % max(1, n_tiles)
        # ~1 in 5 high-attention tiles favours a different subtype.
        dissent = next(stream) < 0.2
        focus = subtype
        if dissent:
            alternatives = [s for s in SUBTYPES if s != subtype]
            focus = alternatives[int(next(stream) * len(alternatives))]

        scores = {}
        for name in SUBTYPES:
            scores[name] = next(stream) * 0.25
        scores[focus] += 1.0
        total = sum(scores.values())
        scores = {k2: v / total for k2, v in scores.items()}

        out.append(
            TileAttribution(
                tile_index=tile_index,
                uri=tile_uri(slide_id, tile_index),
                x=(tile_index % cols) * TILE_PX,
                y=(tile_index // cols) * TILE_PX,
                attention=round(attention, 4),
                subtype_scores={k2: round(v, 4) for k2, v in scores.items()},
            )
        )
        # Monotone decay keeps the list sorted by construction.
        attention *= 0.82 + next(stream) * 0.1

    return out


def classify_slide(
    slide_path,
    slide_id=None,
    force_subtype=None,
    top_k_tiles=5,
    seed=None,
):
    """Classify one whole-slide image. **Stubbed** -- returns fabricated results.

    Signature matches what a real wrapper around the Slideflow ``hcord``
    model would expose, so swapping in real inference does not change any
    call site.

    Args:
        slide_path: Path to the WSI (``.svs``, ``.ndpi``, ...). Only its name
            is used here; the file need not exist unless ``seed`` is None and
            you want path-stable results.
        slide_id: Identifier used in tile URIs. Defaults to the filename stem.
        force_subtype: Testing/demo hook. Force a specific six-class outcome
            (``"A"``, ``"AB"``, ``"B1"``, ``"B2"``, ``"B3"``, ``"TC"``).
            Logits are generated so they *decode* to this, so the real
            decoder still runs. ``None`` means pseudo-random.
        top_k_tiles: How many high-attention tiles to return.
        seed: Overrides the derived seed for full determinism control.

    Returns:
        PathologyPrediction.

    Raises:
        ValueError: If ``force_subtype`` is not a recognised subtype.
    """
    name = os.path.basename(str(slide_path))
    stem = os.path.splitext(name)[0] or "slide"
    resolved_id = slide_id or stem

    stream = _rng_stream(str(seed) if seed is not None else str(slide_path))

    # Plausible bag size: the paper runs every tile, typically thousands.
    n_tiles = 800 + int(next(stream) * 4200)

    logits = _fake_logits(stream, force_subtype=force_subtype)
    coarse_group, coarse_probs, subtype, subtype_conf = decode_logits(logits)

    return PathologyPrediction(
        slide_id=resolved_id,
        logits=[round(v, 4) for v in logits],
        coarse_group=coarse_group,
        coarse_confidence=round(coarse_probs[coarse_group], 4),
        coarse_probs={k: round(v, 4) for k, v in coarse_probs.items()},
        subtype=subtype,
        subtype_confidence=round(subtype_conf, 4),
        n_tiles=n_tiles,
        attributions=_fake_attributions(
            stream, resolved_id, n_tiles, subtype, top_k_tiles
        ),
    )
