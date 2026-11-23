"""Assessment history page."""
import pandas as pd
import streamlit as st

from src.ui import components as ui

ui.setup_page("Assessment History", "🗂️")
ui.hero("Assessment History", "Everything you have analysed, saved on this device so you can look back and compare.")
db = ui.load_db()

kind = st.radio("Show", ["All", "Plants", "Livestock"], horizontal=True)
atype = {"All": None, "Plants": "plant", "Livestock": "livestock"}[kind]
rows = db.list_assessments(limit=200, assessment_type=atype)
if not rows:
    st.info("No assessments yet. Analyse a leaf or assess an animal and it will appear here.", icon="📭")
    st.stop()

df = pd.DataFrame([{
    "ID": r["id"], "Date": r["created_at"].replace("T", " ")[:16], "Type": r["assessment_type"].title(),
    "Result": r["summary"], "Confidence": f"{r['confidence']:.0%}" if r["confidence"] is not None else "-",
    "Photo": "Yes" if r["image_path"] else "No"} for r in rows])
st.dataframe(df, hide_index=True)

st.subheader("Open an assessment")
labels = {r["id"]: f"#{r['id']} - {r['created_at'][:10]} - {r['summary'][:70]}" for r in rows}
chosen = st.selectbox("Choose one", list(labels), format_func=labels.get)
a = db.get_assessment(chosen)
if a:
    c1, c2 = st.columns([1, 2])
    with c1:
        img = ui.image_from_upload_dir(a.get("image_path"))
        if img:
            st.image(str(img), caption="Saved photo", width="stretch")
        else:
            st.caption("No photo saved for this assessment.")
    with c2:
        st.markdown(f"**Type:** {a['assessment_type'].title()}  \n**Date:** {a['created_at'].replace('T', ' ')[:16]} UTC  \n**Result:** {a['summary']}")
        if a.get("crop"):
            st.markdown(f"**Crop:** {a['crop']}")
        if a.get("confidence") is not None:
            st.markdown(f"**Model confidence (calibrated):** {a['confidence']:.0%}  \n**Model version:** {a.get('model_version')}")
        if a.get("notes"):
            st.markdown(f"**Notes:** {a['notes']}")
        if a["symptoms"]:
            st.markdown("**Symptoms reported:** " + "; ".join(s["symptom_label"] for s in a["symptoms"]))
    if a["predictions"]:
        st.plotly_chart(ui.probability_chart([(p["label_name"], p["probability"]) for p in a["predictions"]]), key=f"hist_chart_{chosen}")
    if a["recommendations"]:
        ui.render_recommendations(a["recommendations"])
    st.divider()
    suggestions = ["What does this result mean?", "How can I prevent this?", "Should I see an expert?"]
    ui.render_followups(chosen, "hist", suggestions)
    with st.expander("Delete this assessment"):
        sure = st.checkbox("Yes, delete it permanently", key=f"del_ok_{chosen}")
        if st.button("Delete", disabled=not sure, key=f"del_{chosen}"):
            db.delete_assessment(chosen)
            st.rerun()
