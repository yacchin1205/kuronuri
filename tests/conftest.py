import pymupdf
import pytest

SAMPLE_LINES = [
    "お客様情報",
    "氏名: 山田 太郎 様",
    "住所: 〒100-0001 東京都千代田区千代田1-1",
    "電話: 03-1234-5678 / ０９０－１２３４－５６７８",
    "メール: taro.yamada@example.com",
    "個人番号: 1234 5678 9012",
    "担当: 株式会社サンプル商事 営業部",
    "本日は晴れです。",
]


@pytest.fixture
def sample_pdf(tmp_path):
    path = tmp_path / "sample.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    for i, line in enumerate(SAMPLE_LINES):
        page.insert_text((50, 80 + i * 24), line, fontname="japan", fontsize=12)
    doc.save(path)
    doc.close()
    return path
