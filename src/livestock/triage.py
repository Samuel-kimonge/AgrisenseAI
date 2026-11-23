"""Rule-based livestock triage engine.

The engine evaluates the JSON rules in data/knowledge_base/livestock_rules.json.
It never names a medicine or dose, never claims a diagnosis, and always includes
veterinary referral guidance. The rules are a DRAFT until reviewed by a veterinarian.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from config import get_settings

from .schema import LivestockCase, SYMPTOMS, AGE_RANGES, DURATIONS

URGENCY_ORDER = ["monitor", "soon", "urgent", "emergency"]
URGENCY_LABEL = {
    "monitor": "Monitor closely",
    "soon": "Veterinary check within 1-2 days",
    "urgent": "Contact a vet today",
    "emergency": "Emergency - call a vet now",
}
REFERRAL = {
    "monitor": "If the animal gets worse or others are affected, contact a veterinarian or animal health worker.",
    "soon": "Arrange a veterinary or animal-health-worker visit within a day or two.",
    "urgent": "Contact a veterinarian or animal health worker today.",
    "emergency": "Call a veterinarian immediately. If you cannot reach one, contact your county veterinary office or the nearest animal health clinic.",
}
DISCLAIMER = ("This is a preliminary, rule-based triage guide, not a diagnosis. It cannot replace examination by a "
              "veterinarian, and it does not recommend any medicine or dose. The rules are a draft awaiting "
              "veterinary review.")


@dataclass
class Concern:
    rule_id: str
    name: str
    urgency: str
    why: str
    notifiable: bool
    next_steps: list[str]


@dataclass
class LivestockAssessment:
    urgency: str
    urgency_label: str
    observed: list[str]
    concerns: list[Concern]
    next_steps: list[str]
    general_steps: list[str]
    referral: str
    notifiable_possible: bool
    disclaimer: str
    rules_version: str
    review_status: str
    adjustments: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["concerns"] = [c.__dict__ for c in self.concerns]
        return d


@lru_cache(maxsize=4)
def load_rules(path: str | None = None) -> dict:
    p = Path(path) if path else get_settings().knowledge_base_dir / "livestock_rules.json"
    return json.loads(p.read_text())


def _eval(node: dict, case: LivestockCase) -> bool:
    if "all" in node:
        return all(_eval(n, case) for n in node["all"])
    if "any" in node:
        return any(_eval(n, case) for n in node["any"])
    if "not" in node:
        return not _eval(node["not"], case)
    if "symptom" in node:
        return node["symptom"] in case.symptoms
    if "field" in node:
        val = getattr(case, node["field"], None)
        if val is None:
            return False
        if "in" in node:
            return val in node["in"]
        if "eq" in node:
            return val == node["eq"]
        if "gte" in node:
            return float(val) >= node["gte"]
        if "lte" in node:
            return float(val) <= node["lte"]
    raise ValueError(f"Unsupported rule node: {node}")


def observed_summary(case: LivestockCase) -> list[str]:
    obs = []
    if case.temperature_c is not None:
        obs.append(f"Body temperature: {case.temperature_c:.1f} C")
    for label, val, normal in (("Appetite", case.appetite, "Normal"), ("Activity", case.activity, "Normal"),
                               ("Breathing", case.breathing, "Normal"), ("Movement", case.movement, "Normal")):
        if val != normal:
            obs.append(f"{label}: {val}")
    obs += [SYMPTOMS[s] for s in case.symptoms]
    obs.append(f"Duration: {case.duration}")
    obs.append(f"Group: {case.affected_group}")
    return obs


def assess(case: LivestockCase, rules: dict | None = None) -> LivestockAssessment:
    errs = case.validate()
    if errs:
        raise ValueError("; ".join(errs))
    rules = rules or load_rules()
    matched: list[Concern] = []
    for r in rules["rules"]:
        if _eval(r["when"], case):
            matched.append(Concern(r["id"], r["concern"], r["urgency"], r["why"], bool(r.get("notifiable")), list(r["next_steps"])))
    matched.sort(key=lambda c: -URGENCY_ORDER.index(c.urgency))

    adjustments: list[str] = []
    if matched:
        urgency = matched[0].urgency
    elif case.has_any_signal():
        urgency = "monitor"
        matched = [Concern("nonspecific", "Non-specific signs - no clear pattern matched", "monitor",
                           "The reported signs do not match a specific pattern in the rule set, which does not mean the animal is well.",
                           False, ["Keep observing and recording changes", "Contact a veterinarian if signs persist or worsen"])]
    else:
        urgency = "monitor"
        matched = [Concern("no_signs", "No abnormal signs reported", "monitor",
                           "You did not report any abnormal findings.", False,
                           ["Continue routine observation", "Contact a veterinarian if you notice any change"])]

    # Safety adjustments (only ever escalate)
    idx = URGENCY_ORDER.index(urgency)
    if case.age_range == AGE_RANGES[0] and idx in (0, 1) and case.has_any_signal() and matched[0].rule_id != "no_signs":
        idx += 1
        adjustments.append("Escalated one level because calves can deteriorate quickly.")
    if case.duration in DURATIONS[2:] and idx == 0 and case.has_any_signal() and matched[0].rule_id != "no_signs":
        idx += 1
        adjustments.append("Escalated one level because signs have lasted several days.")
    if case.affected_group == "Several animals affected" and idx < 2 and case.has_any_signal():
        idx = 2
        adjustments.append("Escalated to 'contact a vet today' because several animals are affected (possible infectious cause).")
    urgency = URGENCY_ORDER[idx]

    steps: list[str] = []
    for c in matched:
        for s in c.next_steps:
            if s not in steps:
                steps.append(s)
    return LivestockAssessment(
        urgency=urgency, urgency_label=URGENCY_LABEL[urgency], observed=observed_summary(case), concerns=matched,
        next_steps=steps, general_steps=list(rules["general_steps"]), referral=REFERRAL[urgency],
        notifiable_possible=any(c.notifiable for c in matched), disclaimer=DISCLAIMER,
        rules_version=rules["rules_version"], review_status=rules["review_status"], adjustments=adjustments)
