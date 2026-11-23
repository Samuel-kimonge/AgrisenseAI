"""Plant Disease Detection page."""
import hashlib

import streamlit as st

from src.ui import components as ui
from src.utils.image_quality import InvalidImageError

ui.setup_page("Plant Disease Detection", "🍅")
ui.hero("Plant Disease Detection", "Photograph a tomato leaf and get a preliminary AI assessment with practical guidance.")

predictor = ui.load_predictor()
if not predictor.available:
    st.error("The trained plant model is not available yet, so photos cannot be analysed. "
             "I will not guess. Train the model first (see README):", icon="🚫")
    st.code("python scripts/prepare_dataset.py\npython scripts/train.py", language="bash")
    st.caption(predictor.error or "")

crop = st.selectbox("1. Choose your crop", ["Tomato"], help="More crops will be added in future versions.")

st.markdown("**2. Add a photo of one leaf** - daylight, in focus, leaf filling most of the picture.")
tab_up, tab_cam = st.tabs(["Upload a photo", "Take a photo"])
with tab_up:
    uploaded = st.file_uploader("JPG, JPEG or PNG (up to 8 MB)", type=["jpg", "jpeg", "png"], key="plant_upload")
with tab_cam:
    camera = st.camera_input("Take a photo of the leaf", key="plant_camera")
source = uploaded or camera
data = source.getvalue() if source is not None else None

if data:
    c1, c2 = st.columns([1, 2])
    with c1:
        st.image(data, caption="Your photo", width="stretch")
    with c2:
        notes = st.text_area("Anything else you noticed? (optional)", max_chars=500, key="plant_notes",
                             placeholder="For example: spots started a week ago after heavy rain.")
else:
    notes = ""
    st.info("Add a photo to continue.", icon="📷")

analyze = st.button("3. Analyze leaf", type="primary", disabled=not (predictor.available and data), width="stretch")
digest = hashlib.sha1(data).hexdigest() if data else None

if analyze and data:
    with st.spinner("Analysing your photo..."):
        try:
            outcome = ui.get_service().analyze_plant(data, crop, notes)
            st.session_state["plant_result"] = {"outcome": outcome, "digest": digest}
        except InvalidImageError as exc:
            st.session_state.pop("plant_result", None)
            ui.friendly_error(str(exc))
        except Exception as exc:
            st.session_state.pop("plant_result", None)
            ui.friendly_error(f"Something went wrong while analysing the photo ({type(exc).__name__}). Please try again.")

stored = st.session_state.get("plant_result")
if stored and stored["digest"] == digest:
    outcome = stored["outcome"]
    r = outcome.result
    st.divider()
    st.subheader("Result")
    if r.status == "ok":
        conf = r.confidence or 0
        ui.result_box("ok" if conf >= 0.9 else "warn", f"Likely: {r.predicted_name}",
                      "Preliminary AI assessment - not a confirmed diagnosis.")
        m1, m2, m3 = st.columns(3)
        m1.metric("Model confidence", f"{conf:.0%}", help="Calibrated score. It is not the chance that the diagnosis is correct in your field.")
        m2.metric("Gap to next option", f"{(r.margin or 0):.0%}")
        m3.metric("Photo quality", "Good" if not r.quality.warnings else "Fair")
        st.plotly_chart(ui.probability_chart([(c.name, c.probability) for c in r.top_k]), key="plant_chart")
        for w in r.quality.warnings:
            st.warning(w, icon="📸")
    elif r.status == "uncertain":
        ui.result_box("warn", "I can't tell from this photo", r.message)
        st.markdown("**Why:** " + "; ".join(r.reasons) + ".")
        st.plotly_chart(ui.probability_chart([(c.name, c.probability) for c in r.top_k], "Closest matches (not reliable enough to name)"), key="plant_chart_u")
    else:
        ui.result_box("bad", "This photo can't be analysed", r.message)

    if r.status in ("ok", "uncertain"):
        st.caption("Confidence is the model's calibrated score on photos like its training data (controlled-condition leaf images). "
                   "Real fields can look different, so treat the result as a lead to confirm.")
        ui.render_recommendations(outcome.recommendations)
        if outcome.assessment_id:
            st.success(f"Saved to your history (assessment #{outcome.assessment_id}).", icon="✅")
            st.divider()
            name = r.predicted_name or "this result"
            ui.render_followups(outcome.assessment_id, "plant", [
                "What does this result mean?", "How can I prevent it?", "Should I see an expert?"])
    else:
        ui.render_recommendations(outcome.recommendations)
