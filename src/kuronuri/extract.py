from dataclasses import dataclass

import pymupdf

Rect = tuple[float, float, float, float]


@dataclass(frozen=True)
class Char:
    c: str
    bbox: Rect
    line: int


@dataclass(frozen=True)
class PageText:
    number: int
    text: str
    chars: list[Char | None]  # parallel to text; None for synthesized line breaks


def extract_page(page: pymupdf.Page) -> PageText:
    flags = (pymupdf.TEXTFLAGS_RAWDICT & ~pymupdf.TEXT_PRESERVE_IMAGES) | pymupdf.TEXT_INHIBIT_SPACES
    raw = page.get_text("rawdict", flags=flags)
    text: list[str] = []
    chars: list[Char | None] = []
    line_no = 0
    for block in raw["blocks"]:
        for line in block["lines"]:
            for span in line["spans"]:
                for ch in span["chars"]:
                    text.append(ch["c"])
                    chars.append(Char(ch["c"], tuple(ch["bbox"]), line_no))
            text.append("\n")
            chars.append(None)
            line_no += 1
    return PageText(page.number, "".join(text), chars)


def extract(doc: pymupdf.Document) -> list[PageText]:
    return [extract_page(page) for page in doc]
