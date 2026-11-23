"""Livestock (cattle) Health Assessment page."""
import streamlit as st

from src.livestock import schema as S
from src.livestock.schema import LivestockCase
from src.ui import components as ui
from src.utils.image_quality import InvalidImageError

ui.setup_page("Livestock Health Assessment", "🐄")
ui.hero("Livestock Health Assessment", "Answer a few questions about your cow or calf and get a safe, preliminary triage guide.")
st.warning("This is a triage guide based on symptom rules, not a diagnosis. It never recommends medicines or doses. "
           "If an animal is in serious trouble, call a veterinarian now.", icon="🩺")

with st.form("livestock_form"):
    st.subheader("About the animal")
    c1, c2, c3 = st.columns(3)
    animal_id = c1.text_input("Animal name or tag (optional)", max_chars=60)
    age = c2.selectbox("Age range", S.AGE_RANGES, index=2)
    breed = c3.text_input("Breed (optional)", max_chars=60, placeholder="e.g. Friesian, Ayrshire, Zebu")

    st.subheader("What are you seeing?")
    c1, c2, c3 = st.columns(3)
    appetite = c1.selectbox("Appetite", S.APPETITE)
    activity = c2.selectbox("Activity level", S.ACTIVITY)
    breathing = c3.selectbox("Breathing", S.BREATHING)
    c1, c2 = st.columns(2)
    movement = c1.selectbox("Movement", S.MOVEMENT)
    duration = c2.selectbox("How long has this been going on?", S.DURATIONS)
    group = st.radio("How many animals are affected?", S.GROUP, horizontal=True)

    measured = st.checkbox("I measured the body temperature")
    temp = st.number_input("Body temperature (degrees Celsius)", min_value=30.0, max_value=45.0, value=38.5, step=0.1,
                           help="Adult cattle are normally about 38.0 to 39.3 C. Only used if the box above is ticked.")
    symptoms = st.multiselect("Visible symptoms (choose all that apply)", list(S.SYMPTOMS), format_func=S.SYMPTOMS.get)
    notes = st.text_area("Other notes (optional)", max_chars=1000)
    photo = st.file_uploader("Photo for your records (optional, JPG or PNG)", type=["jpg", "jpeg", "png"])
    st.caption("Photos are stored only as a record. They are not analysed by AI, because there is no validated animal-disease image model in this version.")
    submitted = st.form_submit_button("Assess", type="primary")

if submitted:
    case = LivestockCase(animal_id=animal_id.strip(), age_range=age, breed=breed.strip(), appetite=appetite, activity=activity,
                         breathing=breathing, movement=movement, symptoms=symptoms, duration=duration, affected_group=group,
                         temperature_c=float(temp) if measured else None, notes=notes.strip())
    errs = case.validate()
    if errs:
        for e in errs:
            ui.friendly_error(e)
    else:
        try:
            aid, a, recs = ui.get_service().assess_livestock(case, photo.getvalue() if photo else None)
            st.session_state["livestock_result"] = {"aid": aid, "a": a, "recs": recs}
        except InvalidImageError as exc:
            ui.friendly_error(str(exc))
        except Exception as exc:
            ui.friendly_error(f"Something went wrong ({type(exc).__name__}). Please try again.")

res = st.session_state.get("livestock_result")
if res:
    a, recs, aid = res["a"], res["recs"], res["aid"]
    st.divider()
    st.subheader("Assessment")
    kind = {"monitor": "ok", "soon": "warn", "urgent": "warn", "emergency": "bad"}[a.urgency]
    ui.result_box(kind, a.urgency_label, a.referral)
    if a.notifiable_possible:
        st.error("This pattern may involve a disease that must be reported to the veterinary authority. Isolate the animal and report promptly.", icon="🚨")
    for adj in a.adjustments:
        st.info(adj, icon="⬆️")
    left, right = st.columns(2)
    with left:
        st.markdown("**What you reported**")
        ui.bullets(a.observed)
    with right:
        st.markdown("**Possible health concerns (patterns, not diagnoses)**")
        for c in a.concerns:
            st.markdown(f"- **{c.name}**  \n  <span class='as-small'>{c.why}</span>", unsafe_allow_html=True)
    ui.render_recommendations(recs)
    st.caption(a.disclaimer)
    st.caption(f"Rules version {a.rules_version}. Status: {a.review_status}")
    st.success(f"Saved to your history (assessment #{aid}).", icon="✅")
    st.divider()
    ui.render_followups(aid, "livestock", ["What should I do now?", "How can I prevent this?", "What should I tell the vet?"])
