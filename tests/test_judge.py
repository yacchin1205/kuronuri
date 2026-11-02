import json

import httpx

from kuronuri.detect import Candidate
from kuronuri.judge import Judge


def test_judge_sends_systemone_request_and_parses_answers(monkeypatch):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "model": "llm-qwen3.5-2b",
            "answers": {
                "redact": {"type": "noul", "noul": 0.93},
                "category": {"type": "choice", "choice": "person_name", "confidence": 0.8,
                             "probabilities": {"person_name": 0.8, "not_sensitive": 0.2}},
            },
            "usage": {"input_tokens": 100, "output_tokens": 2},
        })

    judge = Judge("http://judge.test")
    judge.client = httpx.Client(base_url="http://judge.test", transport=httpx.MockTransport(handler))
    cand = Candidate(0, 4, 9, "山田 太郎", "Person", "ner", "氏名: 山田 太郎 様")
    j = judge.judge(cand)

    assert seen["path"] == "/v1/systemone"
    assert seen["body"]["model"] == "jev-latest"
    assert seen["body"]["state"]["marked_text"] == "山田 太郎"
    assert seen["body"]["questions"]["redact"]["type"] == "noul"
    assert seen["body"]["questions"]["category"]["type"] == "choice"
    assert j.redact == 0.93
    assert j.category == "person_name"
    assert j.category_confidence == 0.8
