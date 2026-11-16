"""Explanations: association rules, a rule-based writer, and an optional LLM.

The rule-based writer is the default and is always available. It needs no API
key and no network. Everything it says is traceable to a row in
data/association_rules.csv or a passage in knowledge_base/, both of which carry
source URLs.

The LLM layer is optional. It is given only the structured findings and the
retrieved passages, and is instructed to stay inside them. It never sees a
patient name. If the key is missing, the request fails or the answer looks
unsafe, the rule-based text is used instead and the interface says so.

Limits enforced regardless of layer: no diagnosis, no medicine names, no
dosages, no treatment instructions.
"""

from __future__ import annotations

import csv
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RULES_CSV = ROOT / "data" / "association_rules.csv"

DISCLAIMER = (
    "BloodWise AI is an educational screening and information tool. It is not a "
    "medical diagnostic system and cannot diagnose any condition. Reference "
    "intervals differ between laboratories, and a result outside an interval does "
    "not by itself mean disease. Discuss these results with a qualified healthcare "
    "professional."
)

# Wording that must never appear in generated text: an explicit second-person
# diagnosis, or anything that reads as a dose.
#
# A measurement such as "28 ug/dL" or "under 150 mg/dL" is a concentration, not
# a dose, so a unit followed by "/" is allowed. "65 mg of iron" is not.
_DISEASE = (r"anemia|anaemia|diabetes|cancer|hepatitis|cirrhosis|leukemia|leukaemia|"
            r"thalassemia|thalassaemia|hypothyroidism|hyperthyroidism|"
            r"(?:kidney|liver|heart|thyroid|celiac|coeliac) disease|kidney failure")
BANNED = re.compile(
    r"(?:"
    rf"\byou (?:have|are suffering from|are diagnosed with)\b(?:\W+\w+){{0,4}}?\W+(?:{_DISEASE})\b|"
    r"\byour (?:diagnosis|condition) is\b|"
    r"\bdiagnos(?:is|ed|e)\s+(?:is|as|with)\b|"
    r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|ug|iu|tablets?|capsules?)(?![/\w])|"
    r"\btake\s+\d|\bstart taking\b|\bstop taking\b|"
    r"\bprescrib|\bdosage\b|\bdose of\b"
    r")", re.IGNORECASE)


# ---------------------------------------------------------------- rules ---
@dataclass
class Association:
    rule_id: str
    name: str
    text: str
    sources: list
    matched: list


class AssociationEngine:
    """Evaluates data/association_rules.csv against a set of findings."""

    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else RULES_CSV
        self.rules = []
        with self.path.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                self.rules.append({
                    "id": r["rule_id"], "name": r["name"],
                    "conditions": json.loads(r["conditions_json"]),
                    "text": r["association_text"],
                    "sources": [s.strip() for s in r["source_urls"].split("|") if s.strip()],
                })

    @staticmethod
    def _holds(cond, by_param) -> bool:
        pid, op = cond[0], cond[1]
        f = by_param.get(pid)
        if f is None:
            return False
        status = getattr(f, "status", None)
        value = getattr(f, "value", None)
        if op in ("low", "high", "normal"):
            return status == op
        if op == "between":
            return value is not None and cond[2] <= value <= cond[3]
        if op == "above":
            return value is not None and value >= cond[2]
        if op == "below":
            return value is not None and value <= cond[2]
        return False

    def evaluate(self, findings) -> list:
        by_param = {getattr(f, "parameter"): f for f in findings}
        out = []
        for rule in self.rules:
            if all(self._holds(c, by_param) for c in rule["conditions"]):
                out.append(Association(
                    rule_id=rule["id"], name=rule["name"], text=rule["text"],
                    sources=rule["sources"],
                    matched=[c[0] for c in rule["conditions"]]))
        return out


# ----------------------------------------------------------- explanation ---
@dataclass
class Explanation:
    summary: str
    sections: list = field(default_factory=list)   # {heading, body, sources}
    associations: list = field(default_factory=list)
    next_steps: list = field(default_factory=list)
    citations: list = field(default_factory=list)
    generated_by: str = "rules"
    llm_note: str = ""
    disclaimer: str = DISCLAIMER

    def as_text(self) -> str:
        parts = [self.summary, ""]
        for s in self.sections:
            parts += [s["heading"], s["body"], ""]
        if self.associations:
            parts.append("Possible associations to discuss with a clinician")
            parts += [f"- {a.name}: {a.text}" for a in self.associations]
            parts.append("")
        if self.next_steps:
            parts.append("What may be worth doing next")
            parts += [f"- {s}" for s in self.next_steps]
            parts.append("")
        parts += ["Sources", *[f"- {c}" for c in self.citations], "", self.disclaimer]
        return "\n".join(parts)


def _plural(n, one, many=None):
    return one if n == 1 else (many or one + "s")


class RuleBasedExplainer:
    """Builds the explanation from findings, rules and retrieved passages."""

    def __init__(self, associations: AssociationEngine | None = None):
        self.engine = associations or AssociationEngine()

    def explain(self, findings, passages=None, anomaly=None) -> Explanation:
        passages = passages or []
        abnormal = [f for f in findings if f.status in ("low", "high")]
        unknown = [f for f in findings if f.status == "unknown"]
        normal = [f for f in findings if f.status == "normal"]

        if abnormal:
            names = ", ".join(f"{f.display_name} ({f.status})" for f in abnormal[:6])
            more = f" and {len(abnormal) - 6} more" if len(abnormal) > 6 else ""
            summary = (f"{len(abnormal)} of {len(findings)} {_plural(len(findings), 'result')} "
                       f"fell outside the reference interval used: {names}{more}. "
                       f"{len(normal)} were inside it.")
        else:
            summary = (f"All {len(findings)} {_plural(len(findings), 'result')} with a "
                       f"reference interval fell inside it.")
        if unknown:
            summary += (f" {len(unknown)} {_plural(len(unknown), 'result')} had no verified "
                        f"reference interval and {_plural(len(unknown), 'is', 'are')} "
                        f"reported without a judgement.")

        sections, citations = [], []
        by_param = {}
        for p in passages:
            by_param.setdefault(p.parameter, []).append(p)

        for f in abnormal:
            body = [f"Your result is {f.value:g} {f.unit}; the interval used was "
                    f"{f.range_text()}."]
            for p in by_param.get(f.parameter, []):
                body.append(f"{p.section.capitalize()}: {p.text}")
                cite = p.citation()
                if cite not in citations:
                    citations.append(cite)
            body.append(f.note)
            sections.append({
                "heading": f"{f.display_name} - {f.status}",
                "body": " ".join(x for x in body if x),
                "sources": [p.source_url for p in by_param.get(f.parameter, [])],
            })

        for f in unknown:
            sections.append({
                "heading": f"{f.display_name} - no reference interval",
                "body": (f"Your result is {f.value:g} {f.unit}. {f.note} "
                         f"Compare it with the interval printed on your own report."),
                "sources": [],
            })

        associations = self.engine.evaluate(findings)
        for a in associations:
            for s in a.sources:
                if s not in citations:
                    citations.append(s)

        steps = []
        if abnormal:
            steps.append("Share this report and these results with a doctor, who can read "
                         "them alongside your symptoms and medical history.")
            steps.append("Check the reference interval printed on your own report, since "
                         "laboratories differ.")
        if any(f.issues for f in findings):
            steps.append("Some values were uncertain when read from the file. Confirm them "
                         "against the original report.")
        if unknown:
            steps.append("For results with no stored reference interval, use the interval "
                         "printed on your report.")
        if anomaly is not None and getattr(anomaly, "is_unusual", False):
            steps.append("The combination of values was flagged as uncommon in the training "
                         "population. This is a statistical observation only and is worth "
                         "mentioning to a clinician alongside the individual results.")
        if not steps:
            steps.append("Keep this report for comparison with future tests, and discuss "
                         "anything that concerns you with a doctor.")

        return Explanation(summary=summary, sections=sections, associations=associations,
                           next_steps=steps, citations=citations, generated_by="rules")


# ------------------------------------------------------------------ LLM ---
class LLMProvider:
    """One small interface so another provider can be added in one class."""
    name = "none"

    @property
    def available(self) -> bool:
        return False

    def complete(self, system: str, user: str, timeout: int = 30) -> str:
        raise NotImplementedError


class OpenAIProvider(LLMProvider):
    name = "openai"
    ENDPOINT = "https://api.openai.com/v1/chat/completions"

    def __init__(self, model: str | None = None, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def complete(self, system: str, user: str, timeout: int = 30) -> str:
        payload = json.dumps({
            "model": self.model,
            "temperature": 0.2,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
        }).encode("utf-8")
        req = urllib.request.Request(
            self.ENDPOINT, data=payload,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return body["choices"][0]["message"]["content"]


SYSTEM_PROMPT = """You explain blood test results to someone with no medical training.

Rules you must follow:
- Use ONLY the findings and the reference passages given to you. Do not add medical
  facts from your own knowledge.
- Never diagnose. Never name a disease as something the reader has. Write
  "may be associated with" and "discuss with a doctor".
- Never name a medicine, a supplement dose, or any treatment.
- If the passages do not cover something, say so plainly.
- Plain language, short sentences, no jargon without explaining it.
- End by advising the reader to discuss the results with a healthcare professional.
- Do not invent numbers. Use only the values given.
"""


def build_llm_prompt(findings, passages, associations, anomaly=None) -> str:
    lines = ["FINDINGS", ""]
    for f in findings:
        lines.append(f"- {f.display_name}: {f.value:g} {f.unit} | status: {f.status} | "
                     f"interval used: {f.range_text()} ({f.ref_origin})")
    if associations:
        lines += ["", "RULE-BASED ASSOCIATIONS ALREADY IDENTIFIED", ""]
        lines += [f"- {a.name}: {a.text}" for a in associations]
    if anomaly is not None and getattr(anomaly, "available", False):
        lines += ["", "COMBINATION CHECK", "",
                  f"- {anomaly.message}"]
    lines += ["", "REFERENCE PASSAGES (the only medical facts you may use)", ""]
    for p in passages:
        lines.append(f"[{p.title} / {p.section}] {p.text} (source: {p.source_url})")
    lines += ["", "Write a short, plain-language explanation for the reader."]
    return "\n".join(lines)


class AIExplainer:
    """Rule-based explanation, optionally rewritten by an LLM."""

    def __init__(self, provider: LLMProvider | None = None,
                 rule_explainer: RuleBasedExplainer | None = None):
        self.provider = provider if provider is not None else OpenAIProvider()
        self.rules = rule_explainer or RuleBasedExplainer()

    @property
    def llm_available(self) -> bool:
        return bool(self.provider and self.provider.available)

    def explain(self, findings, passages=None, anomaly=None,
                use_llm: bool = True) -> Explanation:
        base = self.rules.explain(findings, passages=passages, anomaly=anomaly)
        if not use_llm:
            base.llm_note = "AI rewriting was turned off; this is the rule-based text."
            return base
        if not self.llm_available:
            base.llm_note = ("No API key was found, so the rule-based explanation is "
                             "shown. The application is fully usable this way.")
            return base
        try:
            prompt = build_llm_prompt(findings, passages or [], base.associations, anomaly)
            text = self.provider.complete(SYSTEM_PROMPT, prompt)
        except (urllib.error.URLError, urllib.error.HTTPError, KeyError,
                TimeoutError, OSError, ValueError) as exc:
            base.llm_note = (f"The language model could not be reached ({type(exc).__name__}), "
                             f"so the rule-based explanation is shown instead.")
            return base
        if BANNED.search(text):
            base.llm_note = ("The generated text was rejected because it contained wording "
                             "this application does not allow, such as a diagnosis or a "
                             "dose. The rule-based explanation is shown instead.")
            return base
        base.sections.insert(0, {"heading": "In plain language", "body": text.strip(),
                                 "sources": []})
        base.generated_by = f"rules + {self.provider.name}"
        base.llm_note = ("An AI model rewrote the explanation using only the findings and "
                         "the sourced passages below.")
        return base
