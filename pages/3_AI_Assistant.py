"""Conversational agricultural assistant page."""
import streamlit as st

from src.services.assistant_service import assessment_context
from src.ui import components as ui

ui.setup_page("AI Agricultural Assistant", "💬")
ui.hero("AI Agricultural Assistant", "Ask about crops, soil, pests and cattle health. Answers come from a curated knowledge base, with sources.")

db = ui.load_db()
if "chat" not in st.session_state:
    st.session_state["chat"] = []          # [{"role","content","sources"}]
if "chat_assessment" not in st.session_state:
    st.session_state["chat_assessment"] = None

recent = db.list_assessments(limit=15)
options = {None: "No assessment (general questions)"}
for r in recent:
    options[r["id"]] = f"#{r['id']} - {r['created_at'][:10]} - {r['summary'][:60]}"
with st.sidebar:
    st.markdown("### Discuss an assessment")
    st.session_state["chat_assessment"] = st.selectbox(
        "Assessment to discuss", list(options), format_func=options.get, key="chat_assessment_select",
        index=list(options).index(st.session_state["chat_assessment"]) if st.session_state["chat_assessment"] in options else 0)
    if st.button("Clear chat", width="stretch"):
        st.session_state["chat"] = []
        st.rerun()

SUGGESTED = ["What causes early blight in tomatoes?", "How do I prevent tomato leaf diseases?", "How should I care for maize?",
             "What are common warning signs of illness in cattle?", "How can I improve soil fertility?",
             "What does my latest disease prediction mean?"]


def run_question(q: str) -> None:
    assistant = ui.load_assistant()
    sel = st.session_state["chat_assessment"]
    if sel is None and "latest" in q.lower() and ("prediction" in q.lower() or "assessment" in q.lower()):
        sel = recent[0]["id"] if recent else None
        if sel is None:
            st.session_state["chat"] += [{"role": "user", "content": q, "sources": []},
                                         {"role": "assistant", "content": "You do not have any saved assessments yet. Analyse a tomato leaf or assess an animal first, then ask me about it.", "sources": []}]
            return
    ctx = assessment_context(db.get_assessment(sel)) if sel else None
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state["chat"]]
    reply = assistant.ask(q, history, ctx)
    st.session_state["chat"] += [{"role": "user", "content": q, "sources": []},
                                 {"role": "assistant", "content": reply.answer, "sources": reply.sources}]


if not st.session_state["chat"]:
    st.markdown("**Try one of these questions:**")
    cols = st.columns(2)
    for i, q in enumerate(SUGGESTED):
        if cols[i % 2].button(q, key=f"sugg_{i}", width="stretch"):
            with st.spinner("Searching the knowledge base..."):
                run_question(q)
            st.rerun()

for m in st.session_state["chat"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        ui.render_sources(m.get("sources", []))

prompt = st.chat_input("Ask a question, for example: How do I prevent tomato leaf diseases?")
if prompt:
    with st.spinner("Searching the knowledge base..."):
        run_question(prompt)
    st.rerun()
st.caption(ui.llm_mode_caption() + " The assistant gives general guidance, never medicine doses or pesticide rates, and is not a substitute for a veterinarian or extension officer.")
