from collections import Counter, defaultdict
from dataclasses import dataclass

import pymupdf

from .extract import Char, PageText, Rect, extract_page


@dataclass(frozen=True)
class Span:
    id: int
    page: int
    start: int
    end: int


class RedactionError(Exception):
    pass


def span_rects(page: PageText, span: Span) -> list[Rect]:
    by_line: dict[int, pymupdf.Rect] = {}
    for ch in page.chars[span.start : span.end]:
        if ch is None:
            continue
        if ch.line in by_line:
            by_line[ch.line] |= ch.bbox
        else:
            by_line[ch.line] = pymupdf.Rect(ch.bbox)
    if not by_line:
        raise RedactionError(f"candidate {span.id} covers no characters")
    others = [ch for i, ch in enumerate(page.chars) if ch is not None and not span.start <= i < span.end]
    return [tuple(_shrink_away_from(r, others, span)) for r in by_line.values()]


def _shrink_away_from(rect: pymupdf.Rect, others: list[Char], span: Span) -> pymupdf.Rect:
    """MuPDF deletes any glyph a redaction overlaps by more than roughly a tenth of its size, so the
    rect must not intersect glyphs outside the span. Neighbouring glyph boxes routinely overlap
    by a point or two (tight leading, proportional fonts, OCR layers); give up the overlap."""
    for ch in others:
        if not rect.intersects(ch.bbox):
            continue
        b = pymupdf.Rect(ch.bbox)
        moves = [
            (b.x1 - rect.x0, pymupdf.Rect(b.x1, rect.y0, rect.x1, rect.y1)),
            (rect.x1 - b.x0, pymupdf.Rect(rect.x0, rect.y0, b.x0, rect.y1)),
            (b.y1 - rect.y0, pymupdf.Rect(rect.x0, b.y1, rect.x1, rect.y1)),
            (rect.y1 - b.y0, pymupdf.Rect(rect.x0, rect.y0, rect.x1, b.y0)),
        ]
        cost, rect = min(moves, key=lambda m: m[0])
        if rect.is_empty:
            raise RedactionError(
                f"candidate {span.id} overlaps non-target text {ch.c!r} at {tuple(round(v, 1) for v in ch.bbox)}"
                " too much to be redacted without damaging it"
            )
    return rect


def _prepare_input(doc: pymupdf.Document, drop_annotations: bool) -> None:
    if doc.embfile_count():
        raise RedactionError("document has embedded files; remove them first")
    for page in doc:
        if drop_annotations:
            annot = page.first_annot
            while annot:
                annot = page.delete_annot(annot)
            continue
        kinds = sorted({a.type[1] for a in page.annots()})
        if kinds:
            raise RedactionError(
                f"page {page.number + 1} has annotations ({', '.join(kinds)}); "
                "they are outside the text layer and would survive redaction. "
                "Pass --drop-annotations to delete all annotations, or flatten them first"
            )


def redact(doc: pymupdf.Document, spans: list[Span], pages: list[PageText],
           drop_annotations: bool = False) -> dict[int, list[Rect]]:
    _prepare_input(doc, drop_annotations)
    rects: dict[int, list[Rect]] = defaultdict(list)
    for span in spans:
        rects[span.page].extend(span_rects(pages[span.page], span))
    for number, page_rects in rects.items():
        page = doc[number]
        for r in page_rects:
            page.add_redact_annot(r, fill=(0, 0, 0))
        page.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_PIXELS)
    # hidden_text=False: an invisible OCR layer on a scanned page is text like any other;
    # only the detected spans may be removed, and verify() checks exactly that.
    doc.scrub(redactions=False, hidden_text=False)
    return rects


def verify(doc: pymupdf.Document, rects: dict[int, list[Rect]], before: list[PageText]) -> None:
    """Fail unless exactly the characters inside the rects are gone and nothing else moved."""
    for number, page_rects in rects.items():
        page_rects = [pymupdf.Rect(r) for r in page_rects]
        after = extract_page(doc[number])

        def inside(ch: Char) -> bool:
            center = pymupdf.Point((ch.bbox[0] + ch.bbox[2]) / 2, (ch.bbox[1] + ch.bbox[3]) / 2)
            return any(center in r for r in page_rects)

        leaked = [ch.c for ch in after.chars if ch is not None and inside(ch)]
        if leaked:
            raise RedactionError(f"page {number + 1}: text remains under redaction: {''.join(leaked)!r}")
        expected = [(ch.c, _rounded(ch.bbox)) for ch in before[number].chars if ch is not None and not inside(ch)]
        remaining = [(ch.c, _rounded(ch.bbox)) for ch in after.chars if ch is not None]
        if remaining != expected:
            lost = list((Counter(expected) - Counter(remaining)).elements())
            gained = list((Counter(remaining) - Counter(expected)).elements())
            raise RedactionError(
                f"page {number + 1}: text outside redactions changed; "
                f"lost {_describe(lost)}; appeared {_describe(gained)}; redactions {_describe_rects(page_rects)}"
            )


def _describe(chars: list[tuple[str, Rect]]) -> str:
    if not chars:
        return "nothing"
    shown = ", ".join(f"{c!r}@({b[0]},{b[1]},{b[2]},{b[3]})" for c, b in chars[:10])
    return shown + (f" and {len(chars) - 10} more" if len(chars) > 10 else "")


def _describe_rects(rects: list[pymupdf.Rect]) -> str:
    return ", ".join(f"({r.x0:.1f},{r.y0:.1f},{r.x1:.1f},{r.y1:.1f})" for r in rects)


def _rounded(bbox: Rect) -> Rect:
    return tuple(round(v, 1) for v in bbox)
