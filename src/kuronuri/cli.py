import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pymupdf

from .detect import detect
from .extract import extract
from .judge import Judge
from .redact import Span, redact, verify


def main() -> None:
    p = argparse.ArgumentParser(prog="kuronuri", description="Detect and black out sensitive text in a PDF")
    p.add_argument("pdf_in", type=Path)
    p.add_argument("pdf_out", type=Path)
    p.add_argument("--judge", default="http://127.0.0.1:8765", help="local-jev base URL")
    p.add_argument("--model", default="jev-latest")
    p.add_argument("--threshold", type=float, default=0.5, help="redact when P(redact) >= this")
    p.add_argument("--context", type=int, default=60, help="characters of context around a candidate")
    p.add_argument("--ignore", action="append", default=[], metavar="KIND",
                   help="skip candidates of this kind (a regex name such as phone, or a GiNZA label such as Province); repeatable")
    p.add_argument("--report", type=Path, help="write every candidate with its judgement as JSON")
    p.add_argument("--drop-annotations", action="store_true",
                   help="delete all annotations (comments, highlights, stamps) instead of refusing them")
    args = p.parse_args()

    doc = pymupdf.open(args.pdf_in)
    pages = extract(doc)
    judge = Judge(args.judge, args.model)
    records = []
    for page in pages:
        for cand in detect(page, args.context):
            if cand.kind in args.ignore:
                continue
            j = judge.judge(cand)
            approved = j.redact >= args.threshold
            records.append({"id": len(records) + 1, **asdict(cand), "judge": asdict(j), "approved": approved})
            print(f"{'REDACT' if approved else 'keep  '} p{cand.page + 1} {cand.kind:<14} {cand.text!r}"
                  f"  redact={j.redact:.2f} {j.category}", file=sys.stderr)

    spans = [Span(r["id"], r["page"], r["start"], r["end"]) for r in records if r["approved"]]
    rects = redact(doc, spans, pages, args.drop_annotations)
    doc.save(args.pdf_out, garbage=4, deflate=True)
    doc.close()
    verify(pymupdf.open(args.pdf_out), rects, pages)
    if args.report:
        args.report.write_text(json.dumps(
            {"source": str(args.pdf_in), "output": str(args.pdf_out), "candidates": records},
            ensure_ascii=False, indent=1))
    print(f"{len(spans)}/{len(records)} candidates redacted, verified -> {args.pdf_out}", file=sys.stderr)
