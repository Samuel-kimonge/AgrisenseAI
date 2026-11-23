"""Assistant safety layer: restricted-topic detection and output sanitising."""
from __future__ import annotations

import re
from dataclasses import dataclass

FLAG_PATTERNS: dict[str, re.Pattern] = {
    "veterinary_dosing": re.compile(
        r"(\b(dose|dosage|dosing|drench|inject(ion|ing)?|prescri\w+)\b.*\b(cow|cattle|calf|calves|bull|animal|livestock|oxytetracycline|antibiotic|penicillin|ivermectin|albendazole|dewormer|drug|medicine)\b)"
        r"|(\b(antibiotic|antibiotics|dewormer|ivermectin|oxytetracycline|penicillin|albendazole|tetracycline|steroid|painkiller)\b.*\b(how much|how many|dose|dosage|ml|mg|give|administer)\b)"
        r"|(\bhow (much|many)\b.*\b(antibiotic|medicine|drug|dewormer|ivermectin|vaccine|ml|mg)\b)", re.I | re.S),
    "pesticide_rate": re.compile(
        r"(\b(spray|application|mixing|mix|dilution|dilute)\b.*\b(rate|ratio|amount|dose|dosage|concentration)\b)"
        r"|(\bhow (much|many)\b.*\b(pesticide|fungicide|insecticide|herbicide|acaricide|miticide|chemical|spray)\b)"
        r"|(\b(ml|mls|grams?|g|kg)\s*(per|/)\s*(litre|liter|l|hectare|ha|acre|knapsack|20\s*l)\b)"
        r"|(\b(pesticide|fungicide|insecticide|herbicide)\b.*\b(rate|dose|dosage|mix)\b)", re.I | re.S),
    "withdrawal_period": re.compile(r"\b(withdrawal (period|time)|pre-?harvest interval|PHI|milk withhold\w*|safe to (drink|sell|eat).*(after|treat\w*))\b", re.I),
    "serious_clinical": re.compile(
        r"\b(dying|is dead|about to die|can'?t (stand|get up|breathe)|cannot (stand|get up|breathe)|collaps\w+|not breathing|convuls\w+|seizure\w*|bleeding (from|heavily)|emergency|sudden(ly)? died|bloat\w* and (struggling|can'?t))\b", re.I),
}

RESTRICTED_RESPONSES = {
    "veterinary_dosing": ("I can't recommend veterinary medicines or doses. The right medicine and amount depend on the animal's weight, age, "
                          "condition, pregnancy, and the exact product, and mistakes can harm the animal and leave residues in milk and meat. "
                          "Please speak to a veterinarian or qualified animal health worker, who can examine the animal and prescribe safely."),
    "pesticide_rate": ("I can't give pesticide mixing or application rates. Safe rates depend on the specific registered product, crop and stage, "
                       "and the label is the legal authority. Please read the product label and ask your local extension officer or a licensed "
                       "agro-input dealer which registered product suits your problem."),
    "withdrawal_period": ("Withdrawal and pre-harvest periods are product-specific, so I can't state them. They are printed on the product label "
                          "and a veterinarian or extension officer can confirm them. Do not sell or consume milk, meat or produce until you know the period has passed."),
    "serious_clinical": ("This sounds serious. Please contact a veterinarian (or your county veterinary office) or, for a person, a medical "
                         "facility immediately instead of waiting for an online answer."),
}

# Sentence-level patterns that must never appear in generated answers.
_DOSE = re.compile(
    r"\b\d+(\.\d+)?\s*(mg|mcg|ml|mls|cc|g|gm|kg|iu|tablets?|capsules?|litres?|liters?|l)\s*(/|per)\s*"
    r"(kg|l|litre|liter|ha|hectare|acre|head|animal|day|knapsack|20\s*l)\b"
    r"|\bwithdrawal (period|time)\s*(of|is|:)?\s*\d+\s*(days?|hours?|weeks?)\b"
    r"|\b(give|inject|administer|drench|dose|apply|spray)\b[^.]{0,40}\b\d+(\.\d+)?\s*(mg|ml|mls|cc|g|iu)\b", re.I)

SANITISED_NOTE = ("(A sentence containing a specific dose, rate or withdrawal period was removed. I do not provide those; "
                  "please ask a veterinarian, extension officer or read the product label.)")


@dataclass
class SafetyResult:
    flags: list[str]

    @property
    def restricted(self) -> bool:
        return any(f in ("veterinary_dosing", "pesticide_rate", "withdrawal_period") for f in self.flags)

    @property
    def emergency(self) -> bool:
        return "serious_clinical" in self.flags


def classify_query(text: str) -> SafetyResult:
    return SafetyResult([name for name, pat in FLAG_PATTERNS.items() if pat.search(text)])


def restricted_message(flags: list[str]) -> str:
    return "\n\n".join(RESTRICTED_RESPONSES[f] for f in flags if f in RESTRICTED_RESPONSES)


def sanitize_answer(text: str) -> tuple[str, bool]:
    """Remove sentences that contain doses/rates/withdrawal periods. Returns (text, changed)."""
    sentences = re.split(r"(?<=[.!?])\s+|\n", text)
    kept, changed = [], False
    for s in sentences:
        if _DOSE.search(s):
            changed = True
        elif s.strip():
            kept.append(s)
    out = "\n".join(kept) if "\n" in text else " ".join(kept)
    if changed:
        out = out.rstrip() + "\n\n" + SANITISED_NOTE
    return out, changed
