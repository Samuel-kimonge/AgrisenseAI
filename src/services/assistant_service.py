"""Glue between the chat assistant, saved assessments, and the Streamlit session."""
from __future__ import annotations

from config import get_settings
from src.assistant.chat import AgriAssistant, AssistantReply
from src.assistant.llm import get_llm
from src.assistant.retriever import KnowledgeRetriever
from src.database.db import Database, get_database


def build_assistant(settings=None, embeddings=None, llm="auto") -> AgriAssistant:
    settings = settings or get_settings()
    retriever = KnowledgeRetriever(settings, embeddings)
    return AgriAssistant(retriever, get_llm(settings) if llm == "auto" else llm)


def assessment_context(a: dict) -> dict:
    """Turn a stored assessment (Database.get_assessment) into assistant context."""
    if a["assessment_type"] == "plant":
        preds = a["predictions"]
        res = a["result"]
        lines = [f"Crop: {a.get('crop')}.", f"Assessment status: {a['status']}."]
        if a["status"] == "ok" and preds:
            lines.append(f"Model's preliminary prediction: {preds[0]['label_name']} (calibrated confidence {preds[0]['probability']:.0%}). "
                         "This is not a confirmed diagnosis.")
            alt = ", ".join(f"{p['label_name']} ({p['probability']:.0%})" for p in preds[1:])
            if alt:
                lines.append(f"Alternative predictions: {alt}.")
            hint = f"{preds[0]['label_name']} tomato"
        else:
            lines.append("The model was uncertain and did not name a condition. Reasons: " + "; ".join(res.get("reasons", [])) + ".")
            if preds:
                lines.append("Closest matches: " + ", ".join(f"{p['label_name']} ({p['probability']:.0%})" for p in preds) + ".")
            hint = "tomato leaf disease diagnosis uncertain photo"
        if a.get("notes"):
            lines.append(f"Farmer notes: {a['notes']}")
        return {"species": "tomato", "summary": a["summary"], "text": "\n".join(lines), "retrieval_hint": hint}

    res = a["result"]["assessment"]
    case = a["result"]["case"]
    lines = ["Animal: cattle, " + str(case.get("age_range")) + ".",
             f"Triage urgency (rule-based, not a diagnosis): {res['urgency_label']}.",
             "Observed: " + "; ".join(res["observed"]) + ".",
             "Possible concerns (patterns only): " + "; ".join(c["name"] for c in res["concerns"]) + "."]
    return {"species": "cattle", "summary": a["summary"], "text": "\n".join(lines),
            "retrieval_hint": " ".join(c["name"] for c in res["concerns"][:2])}


def ask_about_assessment(assistant: AgriAssistant, db: Database, assessment_id: int, question: str) -> AssistantReply:
    a = db.get_assessment(assessment_id)
    if a is None:
        raise KeyError(f"Assessment {assessment_id} not found")
    history = []
    for f in a["follow_ups"]:
        history += [{"role": "user", "content": f["question"]}, {"role": "assistant", "content": f["answer"]}]
    reply = assistant.ask(question, history, assessment_context(a))
    db.add_follow_up(assessment_id, question, reply.answer, reply.sources)
    return reply
