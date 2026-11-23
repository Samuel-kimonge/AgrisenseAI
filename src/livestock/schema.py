"""Livestock case schema and validation (cattle prototype)."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

AGE_RANGES = ["Calf (under 6 months)", "Young stock (6-24 months)", "Adult (2-8 years)", "Older adult (over 8 years)"]
APPETITE = ["Normal", "Reduced", "Not eating"]
ACTIVITY = ["Normal", "Dull / less active", "Very weak or lying down and not getting up"]
BREATHING = ["Normal", "Fast or shallow breathing", "Laboured breathing"]
MOVEMENT = ["Normal", "Limping / lameness", "Reluctant to move / stiff", "Staggering or circling", "Unable to stand"]
DURATIONS = ["Less than 24 hours", "1-3 days", "4-7 days", "More than 1 week"]
GROUP = ["Only this animal", "Several animals affected"]

SYMPTOMS: dict[str, str] = {
    "coughing": "Coughing",
    "nasal_discharge": "Nasal discharge",
    "eye_discharge": "Eye discharge, watering or cloudiness",
    "diarrhoea": "Diarrhoea (watery dung)",
    "bloody_diarrhoea": "Blood in the dung",
    "bloated_left_side": "Swollen / bloated left side (flank)",
    "drooling": "Drooling or excessive saliva",
    "mouth_sores": "Sores or blisters in the mouth or on the tongue",
    "hoof_lesions": "Sores, blisters or swelling between the hooves",
    "swollen_lymph_nodes": "Swollen lumps in front of the shoulder or below the ear",
    "swollen_joint": "Swollen joint",
    "muscle_swelling": "Hot, painful swelling over a muscle (leg, shoulder, rump)",
    "udder_swelling_hard": "Hard, hot or swollen udder",
    "abnormal_milk": "Abnormal milk (clots, watery, blood)",
    "skin_nodules": "Firm lumps in the skin",
    "skin_lesions_scabs": "Skin scabs or sores",
    "itching_hair_loss": "Itching or hair loss",
    "ticks_present": "Ticks on the animal",
    "red_urine": "Red or brown urine",
    "pale_gums": "Pale gums or eye lining",
    "weight_loss": "Weight loss / poor body condition",
    "stopped_chewing_cud": "Not chewing the cud",
    "abortion": "Aborted a calf",
    "retained_placenta_or_discharge": "Retained placenta or abnormal vaginal discharge",
    "recently_calved_down": "Cow cannot rise after calving",
    "seizures_tremors": "Tremors, fits or seizures",
    "bleeding_from_openings": "Bleeding from nose, mouth, anus or other openings",
    "sudden_death_in_herd": "Sudden death of this or other animals in the herd",
}

MIN_TEMP_C, MAX_TEMP_C = 30.0, 45.0


@dataclass
class LivestockCase:
    species: str = "cattle"
    animal_id: str = ""
    age_range: str = AGE_RANGES[2]
    breed: str = ""
    appetite: str = APPETITE[0]
    activity: str = ACTIVITY[0]
    temperature_c: float | None = None
    breathing: str = BREATHING[0]
    movement: str = MOVEMENT[0]
    symptoms: list[str] = field(default_factory=list)
    duration: str = DURATIONS[0]
    affected_group: str = GROUP[0]
    notes: str = ""
    photo_path: str | None = None

    def validate(self) -> list[str]:
        errs: list[str] = []
        if self.species != "cattle":
            errs.append("Only cattle are supported in this version.")
        for name, value, allowed in (("age range", self.age_range, AGE_RANGES), ("appetite", self.appetite, APPETITE),
                                     ("activity", self.activity, ACTIVITY), ("breathing", self.breathing, BREATHING),
                                     ("movement", self.movement, MOVEMENT), ("duration", self.duration, DURATIONS),
                                     ("group", self.affected_group, GROUP)):
            if value not in allowed:
                errs.append(f"Invalid {name}: {value!r}.")
        unknown = [s for s in self.symptoms if s not in SYMPTOMS]
        if unknown:
            errs.append(f"Unknown symptom(s): {', '.join(unknown)}.")
        if self.temperature_c is not None:
            if not (MIN_TEMP_C <= float(self.temperature_c) <= MAX_TEMP_C):
                errs.append(f"Temperature {self.temperature_c} C is outside the plausible range "
                            f"({MIN_TEMP_C:.0f}-{MAX_TEMP_C:.0f} C). Please check the reading (use degrees Celsius).")
        if len(self.animal_id) > 60 or len(self.breed) > 60:
            errs.append("Animal ID and breed must be 60 characters or fewer.")
        if len(self.notes) > 1000:
            errs.append("Notes must be 1000 characters or fewer.")
        return errs

    def has_any_signal(self) -> bool:
        """True if the farmer reported at least one abnormal finding."""
        return bool(self.symptoms) or self.temperature_c is not None or any([
            self.appetite != APPETITE[0], self.activity != ACTIVITY[0], self.breathing != BREATHING[0],
            self.movement != MOVEMENT[0]])

    def to_dict(self) -> dict:
        return asdict(self)
