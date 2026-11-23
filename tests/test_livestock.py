import json
import re

import pytest

from src.assistant.safety import _DOSE
from src.livestock.schema import LivestockCase, SYMPTOMS
from src.livestock.triage import URGENCY_ORDER, assess, load_rules


def idx(a):
    return URGENCY_ORDER.index(a.urgency)


def test_validation_catches_bad_input():
    assert LivestockCase(temperature_c=5).validate()
    assert LivestockCase(temperature_c=60).validate()
    assert LivestockCase(symptoms=["not_a_symptom"]).validate()
    assert LivestockCase(age_range="Ancient").validate()
    assert LivestockCase(species="goat").validate()
    assert not LivestockCase(temperature_c=38.6, symptoms=["coughing"]).validate()


def test_invalid_case_raises_in_assess():
    with pytest.raises(ValueError):
        assess(LivestockCase(temperature_c=99))


def test_no_signs_is_monitor_and_has_referral():
    a = assess(LivestockCase())
    assert a.urgency == "monitor" and a.concerns[0].rule_id == "no_signs" and a.referral and a.disclaimer


def test_fmd_pattern_is_emergency_and_notifiable():
    a = assess(LivestockCase(symptoms=["mouth_sores", "drooling"], temperature_c=40.2))
    assert a.urgency == "emergency" and a.notifiable_possible
    assert any(c.rule_id == "fmd_pattern" for c in a.concerns)


def test_downer_cow_and_bloat_emergencies():
    assert assess(LivestockCase(symptoms=["recently_calved_down"])).urgency == "emergency"
    assert assess(LivestockCase(symptoms=["bloated_left_side"], breathing="Laboured breathing")).urgency == "emergency"
    assert assess(LivestockCase(symptoms=["bloated_left_side"])).urgency == "urgent"


def test_tick_borne_pattern():
    a = assess(LivestockCase(temperature_c=40.1, symptoms=["swollen_lymph_nodes", "ticks_present"]))
    assert a.urgency == "urgent" and any(c.rule_id == "tick_borne_fever" for c in a.concerns)


def test_mastitis_levels():
    mild = assess(LivestockCase(symptoms=["abnormal_milk"]))
    severe = assess(LivestockCase(symptoms=["udder_swelling_hard"], temperature_c=40.0))
    assert mild.urgency == "soon" and severe.urgency == "urgent"


def test_safety_escalations_only_go_up():
    base = assess(LivestockCase(symptoms=["eye_discharge"]))
    calf = assess(LivestockCase(symptoms=["eye_discharge"], age_range="Calf (under 6 months)"))
    herd = assess(LivestockCase(symptoms=["eye_discharge"], affected_group="Several animals affected"))
    assert idx(calf) > idx(base) and idx(herd) >= 2 and herd.adjustments


def test_low_temperature_emergency():
    assert assess(LivestockCase(temperature_c=36.5)).urgency == "emergency"


def test_every_assessment_includes_referral_and_never_a_dose():
    for sym in SYMPTOMS:
        a = assess(LivestockCase(symptoms=[sym]))
        blob = json.dumps(a.to_dict())
        assert a.referral and a.disclaimer
        assert not _DOSE.search(blob), sym


def test_rules_contain_no_medicine_names_or_doses():
    text = json.dumps(load_rules()).lower()
    for drug in ["oxytetracycline", "penicillin", "ivermectin", "albendazole", "tetracycline", "dexamethasone", "mg/kg"]:
        assert drug not in text
    assert "draft" in load_rules()["review_status"].lower()
