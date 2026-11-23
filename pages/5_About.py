"""About page: purpose, model card summary, safety and limitations."""
import json

import streamlit as st

from config import get_settings
from src.services.knowledge import guide_meta
from src.ui import components as ui

ui.setup_page("About AgriSense AI", "ℹ️")
ui.hero("About AgriSense AI", "An agricultural health prototype: plant photo assessment, cattle triage and a knowledge-grounded assistant.")
s = get_settings()
predictor = ui.load_predictor()

st.markdown("""
**What it does.** AgriSense AI helps farmers (1) check tomato leaf photos for common diseases and pests with a trained model,
(2) work through a structured cattle symptom questionnaire to get a safe triage guide, and (3) ask questions answered from a curated knowledge base.

**What it is not.** It is not a diagnosis and does not replace an extension officer or veterinarian. It never recommends veterinary medicines,
doses, pesticide rates or withdrawal periods.
""")

st.subheader("Plant model")
if predictor.available:
    m = predictor.meta
    st.markdown(f"**Version:** {m['model_version']}  \n**Architecture:** {m['architecture']}  \n**Classes ({len(m['class_keys'])}):** "
                + ", ".join(m["class_names"].values()))
    met = m["metrics"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Held-out test accuracy", f"{met['test_accuracy']:.1%}")
    c2.metric("Macro F1", f"{met['test_macro_f1']:.3f}")
    c3.metric("Calibration error (ECE)", f"{met['test_ece_calibrated']:.3f}")
    c4.metric("Answered / abstained", f"{met['test_coverage']:.0%} / {1 - met['test_coverage']:.0%}")
    st.caption(f"Accuracy among answered photos: {met['test_selective_accuracy']:.1%} (the model abstains when unsure). "
               f"Test images were never used for training, calibration or thresholds. Trained {m['trained_at'][:10]}.")
    ood = met.get("ood_other_crops")
    if ood:
        st.caption(f"Out-of-scope check: on {ood['n']} healthy leaves of other crops, the model abstained on {ood['abstention_rate']:.0%}.")
    cm_path = s.reports_dir / "evaluation" / "confusion_matrix.csv"
    if cm_path.exists():
        import pandas as pd
        import plotly.express as px
        cm = pd.read_csv(cm_path, index_col=0)
        fig = px.imshow(cm.values, x=[ui_n for ui_n in cm.columns], y=list(cm.index), text_auto=True, color_continuous_scale="Greens",
                        labels=dict(x="Predicted", y="Actual", color="Images"), aspect="auto")
        fig.update_layout(height=520, margin=dict(l=10, r=10, t=30, b=10), title="Confusion matrix (held-out test set)")
        with st.expander("Confusion matrix and per-class metrics"):
            st.plotly_chart(fig, key="about_cm")
            rep_path = s.reports_dir / "evaluation" / "classification_report.csv"
            if rep_path.exists():
                st.dataframe(pd.read_csv(rep_path, index_col=0).round(3))
else:
    st.warning("No trained model is installed. Run the training steps in the README.", icon="🚫")
if st.button("Reload model"):
    st.cache_resource.clear()
    st.rerun()

st.subheader("Known limitations")
st.markdown("""
- The model was trained on **PlantVillage**, whose photos are taken on plain backgrounds in controlled conditions. Accuracy on real field photos (clutter, shadows, mixed infections, other varieties) is expected to be **lower** and has **not** been measured. Field validation is needed before real-world reliance.
- Random train/test splits can place near-identical photos of the same leaf on both sides, which can make test accuracy look **better** than it would be on new farms.
- Confidence scores are calibrated on this dataset only. High confidence is not a guarantee.
- Several conditions look alike (for example early blight and target spot; viral diseases, herbicide damage and nutrient problems). Only tomato leaves are supported.
- Livestock triage uses **draft rules** that have not been reviewed by a veterinarian. There is **no** animal image diagnosis.
- The knowledge base is a **draft** compiled from general guidance, pending review by agronomists and veterinarians. It is not a pesticide or drug registry.
- The offline search uses lexical embeddings; it can miss paraphrased questions. Configure neural embeddings or an LLM for better wording.
""")

st.subheader("Knowledge base")
meta = guide_meta()
st.caption(f"Tomato condition guide v{meta['version']} - {meta['review_status']}")
mode = "no language model (extractive answers)" if s.llm_provider in ("", "none") else f"{s.llm_provider} ({s.llm_model or 'default model'})"
st.caption(f"Assistant mode: {mode}. Embeddings: {s.embedding_provider}.")
