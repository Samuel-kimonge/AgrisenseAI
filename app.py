"""AgriSense AI - Streamlit entry point.  Run:  streamlit run app.py"""
import streamlit as st

from src.ui import components as ui

st.set_page_config(page_title="AgriSense AI", page_icon="🌱", layout="wide")

TIPS = [
    "Scout your crop at least once a week and check the undersides of leaves.",
    "Water at the base of tomato plants in the morning so leaves dry quickly.",
    "Rotate tomatoes with non-solanaceous crops for two to three seasons.",
    "Take leaf photos in daylight, in focus, with one leaf filling the frame.",
    "Write down your cattle's normal behaviour and milk yield so you notice change early.",
    "Isolate newly bought animals before mixing them with your herd.",
    "A soil test is the best basis for fertiliser and lime decisions.",
    "Always read the product label and ask an extension officer before using chemicals.",
]


def home() -> None:
    from datetime import date
    ui.setup_page("Home", "🌱")
    ui.hero("Welcome to AgriSense AI", "Check tomato leaves, get cattle health guidance, and ask farming questions - all in one place.")
    db, predictor = ui.load_db(), ui.load_predictor()

    st.subheader("What would you like to do?")
    c1, c2, c3 = st.columns(3)
    with c1:
        ui.card("🍅 Check a tomato leaf", "Upload or take a photo and get a preliminary AI assessment with guidance.")
        st.page_link("pages/1_Plant_Disease.py", label="Open Plant Disease Detection", icon="🍅")
    with c2:
        ui.card("🐄 Check a sick animal", "Answer simple questions and get a safe triage guide for your cattle.")
        st.page_link("pages/2_Livestock_Health.py", label="Open Livestock Health", icon="🐄")
    with c3:
        ui.card("💬 Ask a question", "Crops, soil, pests and cattle care - with sources.")
        st.page_link("pages/3_AI_Assistant.py", label="Open AI Assistant", icon="💬")

    st.subheader("Your farm at a glance")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Assessments", db.count_assessments())
    m2.metric("Plant checks", db.count_assessments("plant"))
    m3.metric("Livestock checks", db.count_assessments("livestock"))
    m4.metric("Plant model", "Ready" if predictor.available else "Not trained")
    st.caption("Supported crops: Tomato.  Supported livestock: Cattle (symptom triage).  More coming in future versions.")
    if not predictor.available:
        st.warning("The plant model has not been trained yet. Photo analysis is disabled until it is (see README).", icon="🚫")

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Recent assessments")
        rows = db.list_assessments(limit=5)
        if not rows:
            st.info("Nothing here yet. Your saved assessments will appear here.", icon="📭")
        for r in rows:
            conf = f" - confidence {r['confidence']:.0%}" if r["confidence"] is not None else ""
            st.markdown(f"**#{r['id']}** {r['created_at'][:10]} - {r['assessment_type'].title()}: {r['summary']}{conf}")
        st.page_link("pages/4_Assessment_History.py", label="See all history", icon="🗂️")
    with right:
        st.subheader("Tip of the day")
        ui.card("🌿 Farming tip", TIPS[date.today().toordinal() % len(TIPS)])
        st.page_link("pages/5_About.py", label="About, accuracy and limitations", icon="ℹ️")


pages = [
    st.Page(home, title="Home Dashboard", icon="🏠", default=True),
    st.Page("pages/1_Plant_Disease.py", title="Plant Disease Detection", icon="🍅"),
    st.Page("pages/2_Livestock_Health.py", title="Livestock Health Assessment", icon="🐄"),
    st.Page("pages/3_AI_Assistant.py", title="AI Agricultural Assistant", icon="💬"),
    st.Page("pages/4_Assessment_History.py", title="Assessment History", icon="🗂️"),
    st.Page("pages/5_About.py", title="About AgriSense AI", icon="ℹ️"),
]
st.navigation(pages).run()
