# kuronuri

Black out sensitive text in PDFs using a local decision model. Nothing leaves your machine.

Candidates are found with regular expressions and Japanese named-entity recognition (GiNZA),
then each candidate is judged in context by a Jev-compatible "System One" model served locally
by [local-jev](https://github.com/amithgc/local-jev). Approved candidates are removed from the
text layer with true redaction (PyMuPDF `apply_redactions`), not just covered with a box, and the
output is re-read to verify that exactly the redacted characters are gone and nothing else moved.

## Usage

Start the judge server (first run downloads the model, about 4.5 GB):

```sh
git clone https://github.com/amithgc/local-jev && cd local-jev
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e .
.venv/bin/local-jev serve --model llm-qwen3.5-2b
```

Then:

```sh
uv sync
uv run kuronuri input.pdf redacted.pdf
```

Options:

- `--threshold 0.5` redact when the judged probability is at least this value
- `--ignore KIND` skip a candidate kind (`phone`, `Province`, ... as printed in the third column); repeatable
- `--drop-annotations` delete all annotations instead of refusing a PDF that has them
- `--report report.json` write every candidate with its judgement
- `--judge URL` judge server, default `http://127.0.0.1:8765`

## Scope and caveats

- Input must have a text layer. Scanned pages with an invisible OCR layer are redacted through that
  layer, image included; pages without any text are left untouched.
- Annotations and embedded files sit outside the text layer and are refused unless dropped.
- Detection is tuned for Japanese documents (phone, postal code, My Number formats, GiNZA labels).
- Recall is not guaranteed. The judge is a small zero-shot model; measure it on your own documents
  before relying on it, and inspect the output.
- PyMuPDF is pinned below 2.0: that release shifts text that follows a redaction on the same line.
