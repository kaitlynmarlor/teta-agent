"""Printed page numbers: citing what the journal printed, not the PDF index.

THE BUG
-------
``build_db.py`` records ``page = i + 1`` -- the *physical* position in the PDF
file. But a clinician citing a paper needs the *printed* page number: the one
on the page, the one in the journal's pagination, the one a colleague can look
up.

For this corpus the two differ by hundreds::

    Marx 2014 ITMIG Consensus   PDF p1  ->  printed p596
    Marx 2015 WHO 4th ed        PDF p2  ->  printed p1384
    Marx 2022 WHO 5th ed        PDF p3  ->  printed p202
    ESMO 2015 Guidelines        PDF p2  ->  printed pv41   (supplement style)

So a citation reading "page 3" is not merely imprecise -- it is *wrong*. No
such page exists in the published article. That is an audit failure: the
reader cannot verify the claim.

THE FIX
-------
Detect the printed number from each page's running header/footer, derive a
constant ``offset`` per document, then record BOTH numbers. We keep the PDF
index because it is how you find the page in the local file, and we add the
printed label because it is how you cite it.

A wrinkle worth knowing about: the ESMO PDF encodes its page digits in a
Private Use Area font range (U+F643-U+F64C) rather than as ASCII digits, so
``extract_text()`` returns ``'v\\uf647\\uf644'`` instead of ``'v41'``. We map
those codepoints back to digits before parsing.

PORTABILITY (ARIAPlatform)
--------------------------
``PageMap`` is corpus-agnostic: it maps physical -> printed labels for any
paginated document. The ARIA reasoning-node ``evidenceRefs`` URI should always
address the *physical* chunk (stable, machine-resolvable), while the printed
label rides along as display metadata for humans. Keep that split -- machine
addresses and human citations have different stability requirements.
"""

import re
from statistics import median

# The ESMO PDF maps digits into a Private Use Area block. U+F643 == "0".
_PUA_ZERO = 0xF643
_PUA_RANGE = range(_PUA_ZERO, _PUA_ZERO + 10)


def decode_pua_digits(text):
    """Translate Private Use Area digit glyphs back into ASCII digits."""
    return "".join(
        str(ord(ch) - _PUA_ZERO) if ord(ch) in _PUA_RANGE else ch for ch in text
    )


# Printed numbers live in running heads/feet: either a bare number
# ("596 Journal of Thoracic Oncology") or a supplement-prefixed one ("v41").
_BARE = re.compile(r"^(\d{2,4})\b")
_PREFIXED = re.compile(r"\b([a-zA-Z])(\d{1,4})\s*$")


def _candidate_lines(page_text):
    """Header and footer lines -- where running page numbers live."""
    lines = [ln.strip() for ln in page_text.splitlines() if ln.strip()]
    return lines[:2] + lines[-3:]


def detect_printed_number(page_text):
    """Find a printed page number on one page.

    Returns:
        ``(prefix, number)`` where prefix is ``""`` for plain numbering and
        e.g. ``"v"`` for supplement pagination, or ``None`` if not found.
    """
    decoded = decode_pua_digits(page_text)
    for line in _candidate_lines(decoded):
        bare = _BARE.match(line)
        if bare:
            return ("", int(bare.group(1)))
        pref = _PREFIXED.search(line)
        if pref:
            return (pref.group(1), int(pref.group(2)))
    return None


class PageMap:
    """Physical PDF page -> printed page label, for one document.

    Rather than trusting any single detection, we collect every page where a
    number was found, compute ``printed - physical`` for each, and take the
    MEDIAN offset. Running heads are noisy (title pages, reference lists, a
    stray "2015" from a citation), and the median ignores those outliers
    whereas a mean would be dragged off by them.
    """

    def __init__(self, offset=0, prefix="", confidence="none", samples=0):
        self.offset = offset
        self.prefix = prefix
        self.confidence = confidence
        self.samples = samples

    @classmethod
    def from_pages(cls, pages):
        """Infer the mapping from ``[{"text": ..., "page": physical}, ...]``."""
        offsets = []
        prefixes = []
        for page in pages:
            found = detect_printed_number(page["text"])
            if found is None:
                continue
            prefix, number = found
            offsets.append(number - page["page"])
            prefixes.append(prefix)

        if not offsets:
            return cls(confidence="none")

        best = int(median(offsets))
        # How many pages agree with the median? That is our confidence.
        agree = sum(1 for off in offsets if off == best)
        prefix = max(set(prefixes), key=prefixes.count) if prefixes else ""

        if agree >= 3 and agree >= len(offsets) * 0.6:
            confidence = "high"
        elif agree >= 2:
            confidence = "low"
        else:
            return cls(confidence="none")

        return cls(offset=best, prefix=prefix, confidence=confidence, samples=agree)

    def printed_label(self, physical_page):
        """Printed label for a physical page, or ``None`` if unmapped."""
        if self.confidence == "none":
            return None
        return f"{self.prefix}{physical_page + self.offset}"

    def describe(self):
        """One-line summary for build logs."""
        if self.confidence == "none":
            return "no printed pagination detected (will cite PDF page)"
        sign = "+" if self.offset >= 0 else ""
        return (
            f"printed = pdf {sign}{self.offset}, prefix={self.prefix or 'none'}, "
            f"confidence={self.confidence} ({self.samples} pages agree)"
        )


def format_page_citation(printed_page, pdf_page):
    """Render the page part of a citation.

    Shows the printed page (what a clinician cites) and keeps the PDF page
    visible in parentheses (how to find it in the local file)::

        p.596 (pdf p.1)
        pdf p.3            <- when no printed pagination was detected
    """
    if printed_page and str(printed_page) != str(pdf_page):
        return f"p.{printed_page} (pdf p.{pdf_page})"
    if printed_page:
        return f"p.{printed_page}"
    return f"pdf p.{pdf_page}"
