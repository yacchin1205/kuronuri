from dataclasses import dataclass

import httpx

from .detect import Candidate

REDACT_QUESTION = {
    "type": "noul",
    "instructions": (
        "The text inside 【 】 in the excerpt must be blacked out before this document "
        "is disclosed to an outside party. Judge only the marked text, not the rest of the excerpt."
    ),
    "criteria": {
        "true": (
            "The marked text identifies or contacts a private individual (name, address, "
            "phone, email, personal ID number, age, health or financial detail), or is "
            "confidential business information such as non-public figures or account numbers."
        ),
        "false": (
            "The marked text is public knowledge, a generic word, a job title, a well-known "
            "place or organization used in a non-sensitive way, or the author's own public identity."
        ),
    },
}

CATEGORY_QUESTION = {
    "type": "choice",
    "instructions": "What kind of information is the text inside 【 】",
    "criteria": {
        "person_name": "Name of a private individual",
        "contact": "Phone number, email address or similar contact detail",
        "id_number": "Personal or account identification number",
        "address": "Postal address or specific location of a private individual",
        "organization": "Company, school or organization name",
        "health": "Medical or health related detail",
        "financial": "Money amount, account or payment detail",
        "not_sensitive": "Not sensitive at all",
        "other": "Sensitive, but none of the above",
    },
}


@dataclass
class Judgement:
    redact: float
    category: str
    category_confidence: float


class Judge:
    def __init__(self, base_url: str, model: str = "jev-latest", timeout: float = 120.0):
        self.model = model
        self.client = httpx.Client(base_url=base_url, timeout=timeout)

    def judge(self, candidate: Candidate) -> Judgement:
        state = {"excerpt": candidate.context, "marked_text": candidate.text}
        response = self.client.post(
            "/v1/systemone",
            json={
                "state": state,
                "model": self.model,
                "questions": {"redact": REDACT_QUESTION, "category": CATEGORY_QUESTION},
            },
        )
        response.raise_for_status()
        answers = response.json()["answers"]
        return Judgement(
            answers["redact"]["noul"],
            answers["category"]["choice"],
            answers["category"]["confidence"],
        )
