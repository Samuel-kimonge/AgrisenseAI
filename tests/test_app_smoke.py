"""Streamlit smoke tests: every page renders without exceptions (model unavailable path)."""
from pathlib import Path

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]

ROOT = Path(__file__).resolve().parents[1]

PAGES = [str(ROOT / "pages" / n) for n in ["1_Plant_Disease.py", "2_Livestock_Health.py", "3_AI_Assistant.py",
                                            "4_Assessment_History.py", "5_About.py"]]


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(isolated_env, page):
    at = AppTest.from_file(str(ROOT / page), default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]


def test_plant_page_says_model_unavailable_and_disables_analysis(isolated_env):
    at = AppTest.from_file(str(ROOT / "pages" / "1_Plant_Disease.py"), default_timeout=60).run()
    assert any("not available" in e.value for e in at.error)
    assert [b for b in at.button if "Analyze" in b.label][0].disabled


def test_home_entrypoint_renders(isolated_env):
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90).run()
    assert not at.exception, [e.value for e in at.exception]


def test_livestock_form_submission(isolated_env):
    at = AppTest.from_file(str(ROOT / "pages" / "2_Livestock_Health.py"), default_timeout=60).run()
    at.multiselect[0].set_value(["mouth_sores", "drooling"])   # in-form values commit on submit, in the same run
    [b for b in at.button if b.label == "Assess"][0].click().run()
    assert not at.exception
    assert any("Emergency - call a vet now" in m.value for m in at.markdown)
    assert any("must be reported" in e.value for e in at.error)
    assert any("Saved to your history" in s.value for s in at.success)


def test_livestock_form_rejects_nothing_reported_gracefully(isolated_env):
    at = AppTest.from_file(str(ROOT / "pages" / "2_Livestock_Health.py"), default_timeout=60).run()
    [b for b in at.button if b.label == "Assess"][0].click().run()
    assert not at.exception and any("Monitor closely" in m.value for m in at.markdown)


def test_assistant_page_answers_question(isolated_env):
    at = AppTest.from_file(str(ROOT / "pages" / "3_AI_Assistant.py"), default_timeout=90).run()
    at.chat_input[0].set_value("How can I improve soil fertility?").run()
    assert not at.exception
    assert any("organic matter" in m.value.lower() for m in at.chat_message[1].markdown)
