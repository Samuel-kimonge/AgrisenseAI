"""Recommendation engine.

Builds structured recommendations from (a) the model result or the triage result,
(b) the curated condition guide, and (c) fixed safety rules. It never names
medicines/doses/pesticide rates and always separates AI output from professional diagnosis.
"""
from __future__ import annotations

from src.livestock.triage import LivestockAssessment
from src.ml.inference import PredictionResult

from .knowledge import get_condition

SECTIONS = ["symptoms", "possible_causes", "prevention", "management", "referral"]

PLANT_DISCLAIMER = ("This is a preliminary AI assessment of one photo, not a confirmed diagnosis. The model was trained on "
                    "controlled-condition leaf photos and can be wrong in real fields. Confirm with a local extension officer "
                    "before taking costly or chemical action.")
GENERAL_REFERRAL = ["Contact your local agricultural extension officer or a licensed agro-input dealer to confirm the problem and to choose "
                    "locally registered products. Follow product labels exactly."]


def plant_recommendations(result: PredictionResult) -> dict[str, list[str]]:
    recs: dict[str, list[str]] = {s: [] for s in SECTIONS}
    recs["referral"] = list(GENERAL_REFERRAL)

    if result.status == "rejected_quality":
        recs["management"] = ["Retake the photo: one leaf, in daylight, in focus, filling most of the frame."]
        if result.quality:
            recs["management"] += list(result.quality.issues)
        return recs

    if result.status == "uncertain":
        recs["symptoms"] = ["The model could not match this leaf to a condition with enough certainty."]
        cands = ", ".join(f"{c.name} ({c.probability:.0%})" for c in result.top_k[:3])
        recs["possible_causes"] = [f"Closest matches, none reliable enough to name: {cands}."]
        recs["prevention"] = ["Scout other plants and the undersides of leaves for spots, mould, webbing or curling.",
                              "Practise good hygiene and avoid wetting leaves."]
        recs["management"] = ["Take two or three more photos of different affected leaves in good light.",
                              "Do not apply chemicals based on this result."]
        recs["referral"] = ["Show the plants or photos to an extension officer, who can diagnose in person."] + GENERAL_REFERRAL
        return recs

    cond = get_condition(result.predicted_key) if result.predicted_key else None
    if not cond:
        recs["management"] = ["No guidance is available in the knowledge base for this class."]
        return recs

    recs["symptoms"] = list(cond["symptoms"]) or ["No disease symptoms expected on a healthy leaf."]
    recs["possible_causes"] = list(cond["possible_causes"]) or ["Not applicable."]
    if cond.get("cause_organism"):
        recs["possible_causes"].insert(0, f"Usual cause: {cond['cause_organism']}.")
    recs["prevention"] = list(cond["prevention"])
    recs["management"] = list(cond["management"])
    recs["referral"] = [f"Get expert help if: {g}." for g in cond["get_help_if"]] + GENERAL_REFERRAL

    if result.confidence is not None and result.confidence < 0.90:
        recs["management"].insert(0, "Confidence is moderate, so treat this as a lead to check, not a fact.")
    if cond.get("confused_with"):
        names = []
        for k in cond["confused_with"]:
            c = get_condition(k)
            names.append(c["name"] if c else k)
        recs["management"].append("This can look like: " + ", ".join(names) + ". A local expert can help tell them apart.")
    return recs


def livestock_recommendations(a: LivestockAssessment) -> dict[str, list[str]]:
    return {
        "symptoms": list(a.observed),
        "possible_causes": [f"{c.name}: {c.why}" for c in a.concerns],
        "prevention": ["Ask your veterinarian for a herd health plan (vaccination, parasite and tick control) suited to your area.",
                       "Isolate new animals before mixing them with the herd and keep housing clean and dry."],
        "management": list(a.next_steps) + list(a.general_steps) + list(a.adjustments),
        "referral": [a.referral] + (["This pattern may involve a notifiable disease: report suspected cases to the veterinary authority."]
                                    if a.notifiable_possible else []),
    }
