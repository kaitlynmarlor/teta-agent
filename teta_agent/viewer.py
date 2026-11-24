"""Clickable citations: open a cited passage in the real PDF, highlighted.

WHY THIS MODULE EXISTS
----------------------
A citation a reader cannot check is a citation they have to take on trust,
and trust is the thing this whole project is trying to avoid needing. The
trace already tells a clinician *which* passage a claim rests on; this
closes the last gap by showing them *the passage itself, on the page it was
printed on*.

HOW THE HIGHLIGHT IS LOCATED -- AND WHY NOT BY LINE NUMBER
----------------------------------------------------------
Each chunk stores ``line_start``/``line_end``, but those are lines **as
pypdf extracted them**, not a property of the PDF. Re-extract with
different settings and they shift. Driving a highlight from them would put
yellow boxes in the wrong place, which is worse than no highlight at all --
it would be a confident, wrong answer about where evidence came from.

So the highlight is found by **matching the rendered PDF's own text lines
against the stored chunk**, using PyMuPDF:

  1. Canonicalise the chunk (lowercase, drop hyphenation and punctuation)
     into one comparison key.
  2. Walk every text line PyMuPDF reports on the page and highlight the
     ones whose canonical form is contained in that key. Whole lines, so a
     sentence wrapping across a line break is covered end to end.
  3. If that yields nothing, fall back to a literal search for the
     build-time anchors. If that also fails, highlight nothing and say so.
     Never guess.

Two earlier versions got this wrong, both in ways that looked plausible:

* Searching for each sentence truncated to 60 characters. ``search_for``
  cannot match across a line break, so the needle usually matched only
  the fragment on one line and the highlight was a scatter of partial
  sentences.
* Testing whole lines with ``line_text in passage``. Better, but it drops
  any line where the chunk starts or ends mid-line, and any line carrying
  a superscript reference the chunk does not reproduce -- so the first
  line was often missing and paragraphs had holes.

Word-level alignment fixes both: it locates where the passage starts in
the page's word stream and walks forward, tolerating the occasional
unmatched word, producing one contiguous run.

Line numbers remain useful as a human-readable address ("p.v48, lines
22-56") and as an audit record. They are just not load-bearing.

PORTABILITY CONTRACT (ARIAPlatform)
-----------------------------------
``locate_passage`` is corpus-agnostic: give it a PDF path, a page and some
text, and it returns rectangles. The HTTP layer is deliberately thin and
dependency-free (``http.server``) so nothing new is required to run it.
"""

import html
import re
import threading
from pathlib import Path
from typing import List

#: Rendering resolution. 130 is legible on a projector without producing
#: multi-megabyte PNGs that make the page feel slow.
RENDER_DPI = 130

#: Minimum run of characters that must match for a page line to count as
#: part of the passage. Short enough to survive a hyphenation or ligature
#: difference mid-line, long enough that a common phrase does not match by
#: accident. Only used by the line-containment fallback.
MIN_OVERLAP_CHARS = 24

#: Shortest common run treated as real alignment evidence rather than
#: coincidence. Twelve canonical characters is roughly two words -- long
#: enough that ordinary English overlap does not register.
MIN_BLOCK_CHARS = 12


def _normalise(text):
    """Collapse whitespace so extraction artefacts do not defeat search."""
    return re.sub(r"\s+", " ", text or "").strip()


def _canonical(text):
    """Aggressively fold text for comparison.

    Lowercases, strips soft hyphens and hyphen-newline breaks, and removes
    everything that is not a letter or digit. Two renderings of the same
    sentence -- one from pypdf at build time, one from PyMuPDF now --
    collapse to the same string, which is the only way to compare them
    reliably.
    """
    text = (text or "").replace("\u00ad", "")
    text = re.sub(r"-\s+", "", text)
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _page_words(page):
    """Every word on the page as ``(canonical_text, rect)``, in reading order.

    Words are the right granularity for alignment: they are small enough
    that a chunk can start or end in the middle of a printed line, and
    PyMuPDF gives each one its own box, so the highlight can stop exactly
    where the passage does.
    """
    try:
        raw = page.get_text("words")
    except Exception:
        return []
    # get_text("words") yields (x0, y0, x1, y1, word, block, line, word_no)
    # already sorted by block/line/position.
    out = []
    for item in raw:
        x0, y0, x1, y1, word = item[0], item[1], item[2], item[3], item[4]
        key = _canonical(word)
        if key:
            out.append((key, (x0, y0, x1, y1)))
    return out


def _align_words(page, passage):
    """Rectangles for the words of ``page`` that make up ``passage``.

    WHY NOT TEST EACH LINE
    ----------------------
    Testing ``line_text in passage`` fails in two common situations, and
    both show up as holes in the highlight:

    * The chunk starts or ends **mid-line**. The page line then contains
      text from outside the chunk, so the containment test fails and the
      whole line is dropped -- which is why the first line of a passage
      was often missing.
    * The line carries a **superscript reference marker or figure
      callout** that the chunk text does not reproduce exactly. One stray
      character fails the whole line, leaving a gap mid-paragraph.

    Aligning at word level removes both failure modes. We find where the
    passage begins in the page's word stream, then walk forward
    accumulating words while they continue to match, tolerating the
    occasional word that does not (a reference numeral, a ligature
    casualty). The result is one contiguous run, so the highlight has no
    holes.

    Args:
        page: A PyMuPDF page.
        passage: Canonical form of the chunk text.

    Returns:
        List of ``(x0, y0, x1, y1)`` rectangles, possibly empty.
    """
    words = _page_words(page)
    if not words:
        return []

    # Build the page's own canonical stream plus an index telling us which
    # word each character came from, so a character offset can be mapped
    # back to rectangles.
    stream_parts = []
    owner = []
    for i, (key, _rect) in enumerate(words):
        stream_parts.append(key)
        owner.extend([i] * len(key))
    stream = "".join(stream_parts)

    # Find the best anchor: the longest prefix of the passage that occurs
    # verbatim in the page stream. Starting from a long, certain match
    # avoids locking onto a coincidental short one.
    start_char = -1
    probe = min(len(passage), 160)
    while probe >= MIN_BLOCK_CHARS:
        start_char = stream.find(passage[:probe])
        if start_char != -1:
            break
        probe -= 8

    if start_char == -1:
        # The passage may begin mid-word (hyphenated line break at the
        # chunk boundary). Try a window further in.
        for offset in range(8, min(len(passage), 120), 8):
            probe_text = passage[offset:offset + 60]
            if len(probe_text) < MIN_BLOCK_CHARS:
                break
            start_char = stream.find(probe_text)
            if start_char != -1:
                break

    if start_char == -1:
        return []

    first_word = owner[start_char]

    # Walk forward through the page words, consuming the passage. A word
    # that does not match is skipped rather than ending the run, because
    # reference superscripts and figure callouts are interleaved into the
    # printed text but absent from the stored chunk.
    consumed = 0
    last_word = first_word
    misses = 0
    for i in range(first_word, len(words)):
        key = words[i][0]
        if consumed >= len(passage):
            break
        if passage.startswith(key, consumed):
            consumed += len(key)
            last_word = i
            misses = 0
            continue
        # Partial match at the tail of the passage (chunk ends mid-word).
        remaining = len(passage) - consumed
        if remaining < len(key) and key.startswith(passage[consumed:]):
            consumed = len(passage)
            last_word = i
            break
        misses += 1
        # Allow a short gap, then give up: beyond this we have drifted off
        # the passage and would highlight unrelated text.
        if misses > 3:
            break

    return [words[i][1] for i in range(first_word, last_word + 1)]


def locate_passage(pdf_path, page_number, chunk_text, metadata=None):
    """Find on-page rectangles covering a chunk's text.

    Args:
        pdf_path: Path to the source PDF.
        page_number: **Physical** 1-based PDF page (not the printed label).
        chunk_text: The stored chunk text.
        metadata: Chunk metadata; ``anchor_head``/``anchor_tail`` are used
            when present.

    Returns:
        ``(rects, how)`` where ``rects`` is a list of
        ``(x0, y0, x1, y1)`` tuples in PDF points and ``how`` describes
        which strategy succeeded -- ``"anchors"``, ``"sentences"`` or
        ``"none"``. Returning the strategy matters: a reader should be able
        to tell an exact span from a best-effort one.
    """
    import fitz

    metadata = metadata or {}
    doc = fitz.open(str(pdf_path))
    try:
        index = max(0, min(int(page_number) - 1, doc.page_count - 1))
        page = doc[index]

        passage = _canonical(chunk_text)
        if not passage:
            return [], "none"

        rects = _align_words(page, passage)
        if rects:
            return _merge_lines(rects), "aligned"

        # Fallback: alignment failed (unusual encoding, heavy ligature
        # use). Try literal search on the build-time anchors, which at
        # least pins the start and end of the passage.
        for key in ("anchor_head", "anchor_tail"):
            needle = _normalise(metadata.get(key, ""))
            if not needle:
                continue
            try:
                hits = page.search_for(needle)
            except Exception:
                hits = []
            for rect in hits:
                rects.append((rect.x0, rect.y0, rect.x1, rect.y1))

        if rects:
            return _merge_lines(rects), "anchors"

        return [], "none"
    finally:
        doc.close()


def _merge_lines(rects, tolerance=3.0):
    """Merge rectangles that sit on the same text line.

    Without this, a highlighted paragraph renders as dozens of separate
    boxes with visible seams. Grouping by vertical position and taking the
    horizontal extent produces one clean band per line.
    """
    if not rects:
        return []
    rows: List[List[float]] = []
    for x0, y0, x1, y1 in sorted(rects, key=lambda r: (r[1], r[0])):
        placed = False
        for row in rows:
            if abs(row[1] - y0) <= tolerance:
                row[0] = min(row[0], x0)
                row[2] = max(row[2], x1)
                row[3] = max(row[3], y1)
                placed = True
                break
        if not placed:
            rows.append([x0, y0, x1, y1])
    return [tuple(r) for r in rows]


def render_page_png(pdf_path, page_number, rects=(), dpi=RENDER_DPI):
    """Render one PDF page to PNG bytes with ``rects`` highlighted.

    The highlight is drawn as a translucent yellow overlay so the text
    underneath stays readable -- a solid fill would hide the very words the
    reader came to check.
    """
    import fitz

    doc = fitz.open(str(pdf_path))
    try:
        index = max(0, min(int(page_number) - 1, doc.page_count - 1))
        page = doc[index]

        for (x0, y0, x1, y1) in rects:
            annot = page.add_highlight_annot(fitz.Rect(x0, y0, x1, y1))
            annot.set_colors(stroke=(1, 0.92, 0.23))
            annot.set_opacity(0.38)
            annot.update()

        pix = page.get_pixmap(dpi=dpi)
        return pix.tobytes("png")
    finally:
        doc.close()


# ── Local HTTP viewer ──────────────────────────────────────────

_PAGE_CSS = """
body { margin:0; background:#1c1f24; color:#e6e6e6;
       font:14px -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; }
header { padding:14px 20px; background:#262a31; border-bottom:1px solid #3a3f47;
         position:sticky; top:0; }
.label { display:inline-block; background:#ffd400; color:#1c1f24;
         font-weight:700; padding:2px 9px; border-radius:4px; margin-right:10px; }
.cite { color:#b9c0c9; }
.how  { float:right; font-size:12px; color:#8d949d; }
.warn { background:#5a3a1a; color:#ffd9a8; padding:8px 20px; font-size:13px; }
main  { padding:20px; text-align:center; }
img   { max-width:100%; box-shadow:0 4px 22px rgba(0,0,0,.6); background:#fff; }
.passage { max-width:860px; margin:18px auto; text-align:left; background:#22262c;
           border-left:3px solid #ffd400; padding:12px 16px; line-height:1.55;
           color:#cdd3da; font-size:13px; border-radius:0 5px 5px 0; }
.nav { margin:14px 0; }
.nav a { color:#8ab4f8; text-decoration:none; margin:0 10px; }
"""


def build_html(ref, how, rects, img_src):
    """One citation's page view."""
    note = {
        "aligned": "passage aligned word by word",
        "lines": "passage matched line by line",
        "anchors": "located by anchor match (partial)",
        "none": "could not locate text on this page",
    }.get(how, how)

    warn = ""
    if how == "none":
        warn = (
            "<div class='warn'>⚠️ The passage text could not be matched on "
            "this page, so nothing is highlighted. The page shown is the one "
            "recorded for this chunk.</div>"
        )

    lines = ""
    if ref.get("line_start"):
        lines = f" · lines {ref['line_start']}–{ref['line_end']}"

    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>{html.escape(ref['label'])} — {html.escape(ref['source'])}</title>
<style>{_PAGE_CSS}</style></head><body>
<header>
  <span class="how">{html.escape(note)} · {len(rects)} region(s)</span>
  <span class="label">{html.escape(ref['label'])}</span>
  <span class="cite">{html.escape(ref['citation'])}{html.escape(lines)}</span>
</header>
{warn}
<main>
  <img src="{img_src}" alt="page">
  <div class="passage"><b>Stored passage text</b><br>{html.escape(ref['text'])}</div>
</main>
</body></html>"""


class CitationViewer:
    """Serves cited passages as highlighted PDF pages over localhost.

    Runs on a daemon thread so the chat loop stays interactive. Started
    lazily on the first ``open`` command -- no server is bound unless the
    feature is actually used.

    Args:
        ledger: The session ``EvidenceLedger``; label -> ref resolution.
        pdf_dir: Directory holding the source PDFs. A chunk id is
            ``{filename}::{index}``, so the filename half locates the file.
        port: TCP port on localhost.
    """

    def __init__(self, ledger, pdf_dir, port=8731):
        self.ledger = ledger
        self.pdf_dir = Path(pdf_dir)
        self.port = port
        self._httpd = None
        self._thread = None

    # -- resolution -------------------------------------------------

    def pdf_for(self, chunk_id, metadata=None):
        """Map a chunk back to its source PDF path.

        Prefers the ``source_file`` metadata field written by
        ``build_db.py``. Falls back to splitting the chunk id, which keeps
        stores built before that field existed working.

        Args:
            chunk_id: ``{filename}::{index}``.
            metadata: Optional chunk metadata.

        Returns:
            ``Path`` to the PDF, or ``None`` if it is not on disk.
        """
        filename = (metadata or {}).get("source_file")
        if not filename:
            filename = chunk_id.split("::")[0]
        path = self.pdf_dir / filename
        return path if path.exists() else None

    def url_for(self, label):
        return f"http://127.0.0.1:{self.port}/p/{label}"

    # -- server -----------------------------------------------------

    def start(self):
        """Start the server if it is not already running. Idempotent."""
        if self._httpd is not None:
            return self.port

        from http.server import BaseHTTPRequestHandler, HTTPServer

        viewer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass  # keep the chat transcript clean

            def _send(self, code, body, ctype):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                try:
                    viewer._route(self)
                except Exception as exc:  # pragma: no cover
                    msg = f"<pre>viewer error: {html.escape(str(exc))}</pre>"
                    self._send(500, msg.encode(), "text/html; charset=utf-8")

        self._httpd = HTTPServer(("127.0.0.1", self.port), Handler)
        self._thread = threading.Thread(
            target=self._httpd.serve_forever, daemon=True
        )
        self._thread.start()
        return self.port

    def stop(self):
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd = None

    def _route(self, handler):
        """Dispatch ``/p/<label>`` (page) and ``/img/<label>`` (PNG)."""
        path = handler.path.split("?")[0]
        parts = [p for p in path.split("/") if p]

        if len(parts) == 2 and parts[0] in ("p", "img"):
            label = parts[1]
            ref = self.ledger.get(label)
            if ref is None:
                body = f"<h2>Unknown citation {html.escape(label)}</h2>"
                handler._send(404, body.encode(), "text/html; charset=utf-8")
                return

            pdf = self.pdf_for(ref.chunk_id)
            if pdf is None:
                body = f"<h2>PDF not found for {html.escape(ref.chunk_id)}</h2>"
                handler._send(404, body.encode(), "text/html; charset=utf-8")
                return

            meta = {
                "anchor_head": getattr(ref, "anchor_head", ""),
                "anchor_tail": getattr(ref, "anchor_tail", ""),
            }
            rects, how = locate_passage(pdf, ref.page, ref.text, meta)

            if parts[0] == "img":
                png = render_page_png(pdf, ref.page, rects)
                handler._send(200, png, "image/png")
                return

            info = {
                "label": ref.label,
                "source": ref.source,
                "citation": ref.citation(),
                "text": ref.text,
                "line_start": getattr(ref, "line_start", 0),
                "line_end": getattr(ref, "line_end", 0),
            }
            page = build_html(info, how, rects, f"/img/{label}")
            handler._send(200, page.encode(), "text/html; charset=utf-8")
            return

        handler._send(404, b"<h2>Not found</h2>", "text/html; charset=utf-8")

    # -- convenience -------------------------------------------------

    def open_label(self, label):
        """Start the server if needed and open ``label`` in the browser.

        Returns:
            ``(ok, message)``.
        """
        ref = self.ledger.get(label)
        if ref is None:
            return False, f"No citation {label} in this session."
        if self.pdf_for(ref.chunk_id) is None:
            return False, f"Source PDF missing for {ref.chunk_id}."

        import webbrowser

        self.start()
        url = self.url_for(label)
        webbrowser.open(url)
        return True, url
