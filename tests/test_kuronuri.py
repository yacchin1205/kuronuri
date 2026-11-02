import pymupdf
import pytest

from kuronuri.detect import detect, detect_regex
from kuronuri.extract import extract
from kuronuri.redact import RedactionError, Span, redact, verify


def test_extract_offsets_map_to_chars(sample_pdf):
    page = extract(pymupdf.open(sample_pdf))[0]
    assert len(page.text) == len(page.chars)
    start = page.text.index("山田")
    assert page.chars[start].c == "山"
    assert page.chars[start + 1].c == "田"
    assert page.chars[start].line == page.chars[start + 1].line


def test_regex_finds_structured_pii(sample_pdf):
    page = extract(pymupdf.open(sample_pdf))[0]
    found = {(c.kind, c.text) for c in detect_regex(page, 10)}
    assert ("email", "taro.yamada@example.com") in found
    assert ("phone", "03-1234-5678") in found
    assert ("phone", "０９０－１２３４－５６７８") in found
    assert ("postal", "〒100-0001") in found
    assert ("mynumber", "1234 5678 9012") in found


def test_ner_finds_person_and_company(sample_pdf):
    page = extract(pymupdf.open(sample_pdf))[0]
    found = {(c.kind, c.text) for c in detect(page)}
    assert ("Person", "山田 太郎") in found or ("Person", "山田") in found
    assert any(kind == "Company" for kind, _ in found)


def test_redact_removes_text_and_verifies(sample_pdf, tmp_path):
    doc = pymupdf.open(sample_pdf)
    before = extract(doc)
    page = before[0]
    targets = ["山田 太郎", "taro.yamada@example.com", "1234 5678 9012"]
    spans = [Span(i, 0, page.text.index(t), page.text.index(t) + len(t)) for i, t in enumerate(targets)]
    rects = redact(doc, spans, before)
    out = tmp_path / "out.pdf"
    doc.save(out, garbage=4, deflate=True)
    doc.close()

    redacted = pymupdf.open(out)
    verify(redacted, rects, before)
    text = redacted[0].get_text()
    for t in targets:
        assert t not in text
    assert "本日は晴れです。" in text
    assert "氏名: " in text


def test_verify_detects_fake_redaction(sample_pdf):
    doc = pymupdf.open(sample_pdf)
    before = extract(doc)
    page = before[0]
    start = page.text.index("山田 太郎")
    span = Span(1, 0, start, start + 5)
    rect = page.chars[start].bbox
    with pytest.raises(RedactionError, match="text remains"):
        verify(doc, {0: [rect]}, before)


def test_annotations_are_refused_unless_dropped(sample_pdf, tmp_path):
    doc = pymupdf.open(sample_pdf)
    doc[0].add_text_annot((300, 300), "reviewer note")
    doc[0].add_highlight_annot(pymupdf.Rect(50, 70, 150, 90))
    annotated = tmp_path / "annotated.pdf"
    doc.save(annotated)

    doc = pymupdf.open(annotated)
    before = extract(doc)
    with pytest.raises(RedactionError, match=r"page 1 has annotations \(Highlight, Text\)"):
        redact(doc, [], before)

    doc = pymupdf.open(annotated)
    before = extract(doc)
    redact(doc, [], before, drop_annotations=True)
    assert next(doc[0].annots(), None) is None


def test_candidate_is_trimmed_and_blank_spans_dropped():
    from kuronuri.detect import _candidate
    from kuronuri.extract import PageText

    page = PageText(0, "氏名: 山田 \n \n様 林 \n", [None] * 15)
    c = _candidate(page, 4, 10, "Person", "ner", 60)
    assert (c.start, c.end, c.text) == (4, 6, "山田")
    assert _candidate(page, 11, 14, "Person", "ner", 60).text == "林"
    assert _candidate(page, 13, 15, "Person", "ner", 60) is None


def test_context_marks_the_candidate():
    from kuronuri.detect import _candidate
    from kuronuri.extract import PageText

    page = PageText(0, "氏名: 山田 太郎 様\n住所: 東京", [None] * 17)
    c = _candidate(page, 4, 9, "Person", "ner", 5)
    assert c.context == "氏名: 【山田 太郎】 様 住所"


def test_invisible_ocr_layer_is_redacted_but_otherwise_kept(sample_pdf, tmp_path):
    pix = pymupdf.open(sample_pdf)[0].get_pixmap(dpi=100)
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_image(page.rect, pixmap=pix)
    page.insert_text((50, 100), "氏名: 山田 太郎 様", fontname="japan", fontsize=12, render_mode=3)
    scanned = tmp_path / "scanned.pdf"
    doc.save(scanned)

    doc = pymupdf.open(scanned)
    before = extract(doc)
    start = before[0].text.index("山田 太郎")
    rects = redact(doc, [Span(1, 0, start, start + 5)], before)
    out = tmp_path / "out.pdf"
    doc.save(out, garbage=4, deflate=True)
    doc.close()

    redacted = pymupdf.open(out)
    verify(redacted, rects, before)
    assert "氏名: " in redacted[0].get_text()
    assert "山田" not in redacted[0].get_text()


def test_redaction_rect_spares_overlapping_neighbours(tmp_path):
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 100), "上の行です", fontname="japan", fontsize=12)
    page.insert_text((50, 112), "山田太郎様", fontname="japan", fontsize=12)   # leading 12pt: boxes overlap 2.4pt
    page.insert_text((50, 124), "下の行です", fontname="japan", fontsize=12)
    tight = tmp_path / "tight.pdf"
    doc.save(tight)

    doc = pymupdf.open(tight)
    before = extract(doc)
    start = before[0].text.index("山田太郎")
    rects = redact(doc, [Span(1, 0, start, start + 4)], before)
    out = tmp_path / "out.pdf"
    doc.save(out, garbage=4, deflate=True)
    doc.close()

    redacted = pymupdf.open(out)
    verify(redacted, rects, before)
    text = redacted[0].get_text()
    assert "上の行です" in text and "下の行です" in text and "様" in text
    assert "山田" not in text
