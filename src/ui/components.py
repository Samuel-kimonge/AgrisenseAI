"""Shared Streamlit UI components: theme, cards, charts, cached resources."""
from __future__ import annotations

import html
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st
from streamlit.errors import StreamlitAPIException

from config import get_settings

GREEN, GREEN_DARK, AMBER, RED = "#2E7D32", "#1B5E20", "#F9A825", "#C62828"

CSS = f"""
<style>
:root {{ --as-green:{GREEN}; --as-dark:{GREEN_DARK}; --as-amber:{AMBER}; --as-red:{RED}; }}
html, body, [class*="css"] {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }}
.block-container {{ padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1100px; }}
h1, h2, h3 {{ letter-spacing: -0.01em; }}
.as-hero {{ background: linear-gradient(135deg, {GREEN_DARK}, {GREEN} 60%, #66BB6A); color: #fff; border-radius: 18px;
  padding: 1.6rem 1.6rem 1.4rem; margin-bottom: 1.2rem; }}
.as-hero h1 {{ color:#fff; margin:0 0 .3rem 0; font-size: 1.9rem; }}
.as-hero p {{ margin:0; opacity:.95; font-size:1.05rem; }}
.as-card {{ border:1px solid rgba(46,125,50,.28); background: rgba(46,125,50,.07); border-radius: 14px; padding: 1rem 1.1rem; margin-bottom:.8rem; }}
.as-card h4 {{ margin:0 0 .25rem 0; }}
.as-result {{ border-left: 6px solid var(--as-green); border-radius: 12px; padding: .9rem 1.1rem; background: rgba(46,125,50,.09); margin: .6rem 0 1rem; }}
.as-result.warn {{ border-left-color: var(--as-amber); background: rgba(249,168,37,.13); }}
.as-result.bad  {{ border-left-color: var(--as-red); background: rgba(198,40,40,.10); }}
.as-result .big {{ font-size: 1.45rem; font-weight: 700; margin: 0; }}
.as-pill {{ display:inline-block; padding:.15rem .65rem; border-radius:999px; font-size:.82rem; font-weight:600; color:#fff; }}
.as-small {{ font-size:.88rem; opacity:.8; }}
.stButton>button, .stDownloadButton>button {{ border-radius: 10px; min-height: 2.7rem; font-weight:600; }}
@media (max-width: 640px) {{ .as-hero h1 {{ font-size:1.5rem; }} .block-container {{ padding-left:.9rem; padding-right:.9rem; }} }}
</style>
"""

URGENCY_COLORS = {"monitor": GREEN, "soon": "#9E9D24", "urgent": "#EF6C00", "emergency": RED}


def setup_page(title: str, icon: str = "🌱") -> None:
    try:
        st.set_page_config(page_title=f"{title} - AgriSense AI", page_icon=icon, layout="wide", initial_sidebar_state="auto")
    except StreamlitAPIException:
        pass  # already set by the entry point
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str) -> None:
    st.markdown(f'<div class="as-hero"><h1>{html.escape(title)}</h1><p>{html.escape(subtitle)}</p></div>', unsafe_allow_html=True)


def card(title: str, body: str) -> None:
    st.markdown(f'<div class="as-card"><h4>{html.escape(title)}</h4><div>{html.escape(body)}</div></div>', unsafe_allow_html=True)


def result_box(kind: str, headline: str, sub: str = "") -> None:
    cls = {"ok": "", "warn": "warn", "bad": "bad"}.get(kind, "")
    st.markdown(f'<div class="as-result {cls}"><p class="big">{html.escape(headline)}</p>'
                f'<div>{html.escape(sub)}</div></div>', unsafe_allow_html=True)


def pill(text: str, color: str) -> str:
    return f'<span class="as-pill" style="background:{color}">{html.escape(text)}</span>'


def probability_chart(items: list[tuple[str, float]], title: str = "Calibrated model confidence") -> go.Figure:
    items = list(reversed(items))
    fig = go.Figure(go.Bar(x=[p * 100 for _, p in items], y=[n for n, _ in items], orientation="h",
                           marker_color=[GREEN if i == len(items) - 1 else "#A5D6A7" for i in range(len(items))],
                           text=[f"{p:.0%}" for _, p in items], textposition="outside", cliponaxis=False))
    fig.update_layout(title=title, xaxis=dict(range=[0, 110], title="Confidence (%)"), yaxis_title=None,
                      margin=dict(l=10, r=30, t=45, b=10), height=60 + 55 * len(items), showlegend=False)
    return fig


def bullets(items: list[str]) -> None:
    for it in items:
        st.markdown(f"- {it}")


def render_recommendations(recs: dict[str, list[str]]) -> None:
    labels = {"symptoms": "Symptoms", "possible_causes": "Possible causes", "prevention": "Prevention",
              "management": "Management guidance", "referral": "Professional referral"}
    tabs = st.tabs([labels[k] for k in labels])
    for tab, key in zip(tabs, labels):
        with tab:
            items = recs.get(key) or []
            bullets(items) if items else st.caption("Nothing to show for this section.")


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for s in sources:
            st.markdown(f"**{s.get('title')}** - section *{s.get('section')}*  \n"
                        f"<span class='as-small'>{s.get('publisher')} | {s.get('publication_date')} | v{s.get('version')} | "
                        f"{s.get('jurisdiction')} | status: {s.get('review_status')}</span>", unsafe_allow_html=True)


def friendly_error(msg: str) -> None:
    st.error(msg, icon="⚠️")


# ----------------------------------------------------------------------- resources
@st.cache_resource(show_spinner="Loading the plant model...")
def load_predictor():
    from src.ml.inference import PlantDiseasePredictor
    return PlantDiseasePredictor()


@st.cache_resource(show_spinner="Preparing the knowledge base...")
def load_assistant():
    from src.services.assistant_service import build_assistant
    return build_assistant()


@st.cache_resource
def load_db():
    from src.database.db import get_database
    return get_database()


def get_service():
    from src.services.assessment_service import AssessmentService
    return AssessmentService(db=load_db(), predictor=load_predictor())


def llm_mode_caption() -> str:
    s = get_settings()
    if s.llm_provider in ("", "none"):
        return "Answers are excerpts from the curated knowledge base (no language model configured)."
    return f"Answers are written by {s.llm_provider} from knowledge-base excerpts."


def render_followups(assessment_id: int, key_prefix: str, suggestions: list[str]) -> None:
    """Follow-up Q&A about one saved assessment."""
    from src.services.assistant_service import ask_about_assessment
    st.subheader("Ask a follow-up question")
    db = load_db()
    a = db.get_assessment(assessment_id)
    for f in (a or {}).get("follow_ups", []):
        with st.chat_message("user"):
            st.write(f["question"])
        with st.chat_message("assistant"):
            st.markdown(f["answer"])
            render_sources(f.get("sources", []))
    cols = st.columns(len(suggestions))
    clicked = None
    for i, (col, q) in enumerate(zip(cols, suggestions)):
        if col.button(q, key=f"{key_prefix}_sugg_{assessment_id}_{i}", width="stretch"):
            clicked = q
    with st.form(f"{key_prefix}_form_{assessment_id}", clear_on_submit=True):
        typed = st.text_input("Your question", placeholder="Type your question here", max_chars=500)
        sent = st.form_submit_button("Ask", type="primary")
    question = clicked or (typed.strip() if sent and typed.strip() else None)
    if question:
        with st.spinner("Looking through the knowledge base..."):
            try:
                ask_about_assessment(load_assistant(), db, assessment_id, question)
            except Exception as exc:
                friendly_error(f"Sorry, I could not answer that right now ({type(exc).__name__}). Please try again.")
                return
        st.rerun()
    st.caption(llm_mode_caption())


def image_from_upload_dir(rel: str | None):
    from src.utils.storage import resolve_image
    return resolve_image(get_settings().uploads_dir, rel)
