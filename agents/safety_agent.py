"""Safety Agent: blocks requests that would cause real harm, before any generation happens.

Layer 1 = fast rules. Layer 2 = LLM check (when a model is reachable) for real, identifiable people.
"""
import re

from agents import llm

MINOR = r"(?:\b(?:child|kid|minor|underage|teen(?:ager)?|schoolgirl|schoolboy|loli\w*)\b|\b(?:[0-9]|1[0-7])[- ]?(?:year|yr)s?[- ]?old\b)"
SEXUAL = r"\b(?:nude|naked|sexual|sex|erotic|porn|lingerie|topless)\b"
RULES = [
    (MINOR + r".*" + SEXUAL, "sexual content involving minors"),
    (SEXUAL + r".*" + MINOR, "sexual content involving minors"),
    (r"\b(fake|forged|counterfeit)\b.*\b(passport|id card|identity card|driver'?s licen[sc]e|visa|certificate|banknote|currency|prescription)\b", "forged documents"),
    (r"\b(bomb|explosive|weapon)\b.*\b(instructions?|how to (make|build))\b", "weapon-making instructions"),
]
REAL_PERSON_PROMPT = (
    "You are a strict safety classifier for an image/video generator. Reply JSON only: "
    '{"block": true|false, "reason": "short"}. Set block=true if the request: (a) depicts a real, named or clearly '
    "identifiable person (celebrity, politician, private individual) in a realistic image/video, especially in a "
    "compromising, criminal, political, sexual or fake-news scene; (b) is sexual/nude content of any real person; "
    "(c) sexualises minors; (d) is meant to deceive (fake evidence, fake documents, fake news footage). "
    "Generic people, fictional characters of your own invention, landscapes, products, animals and normal "
    "portraits of invented people are allowed."
)


class SafetyAgent:
    name = "SafetyAgent"

    def check(self, text: str):
        """Returns (allowed, reason)."""
        t = (text or "").lower()
        for pattern, why in RULES:
            if re.search(pattern, t):
                return False, f"Blocked: {why}."
        try:
            res = llm.parse_json(llm.chat(REAL_PERSON_PROMPT, text, json_mode=True, temperature=0))
            if res.get("block"):
                return False, f"Blocked: {res.get('reason', 'could cause harm')}."
        except Exception:
            pass  # rules still applied; classifier unavailable
        return True, ""
