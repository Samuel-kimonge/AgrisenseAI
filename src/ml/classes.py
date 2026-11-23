"""Class registry for the tomato disease classifier.

Labels are *discovered* from the dataset folders (see dataset.discover_dataset);
this module only maps raw folder names to stable keys and farmer-friendly names.
"""
from __future__ import annotations

# folder name in PlantVillage -> (stable key, display name)
TOMATO_FOLDER_MAP: dict[str, tuple[str, str]] = {
    "Tomato___healthy": ("tomato_healthy", "Healthy tomato leaf"),
    "Tomato___Early_blight": ("tomato_early_blight", "Early blight"),
    "Tomato___Late_blight": ("tomato_late_blight", "Late blight"),
    "Tomato___Bacterial_spot": ("tomato_bacterial_spot", "Bacterial spot"),
    "Tomato___Leaf_Mold": ("tomato_leaf_mold", "Leaf mold"),
    "Tomato___Septoria_leaf_spot": ("tomato_septoria_leaf_spot", "Septoria leaf spot"),
    "Tomato___Spider_mites Two-spotted_spider_mite": ("tomato_spider_mites", "Spider mites (two-spotted)"),
    "Tomato___Target_Spot": ("tomato_target_spot", "Target spot"),
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": ("tomato_yellow_leaf_curl_virus", "Yellow leaf curl virus"),
    "Tomato___Tomato_mosaic_virus": ("tomato_mosaic_virus", "Tomato mosaic virus"),
}

SUPPORTED_CROPS = ["Tomato"]


def key_for_folder(folder: str) -> tuple[str, str]:
    if folder in TOMATO_FOLDER_MAP:
        return TOMATO_FOLDER_MAP[folder]
    # Unknown folder: derive a key rather than silently dropping the class.
    key = folder.lower().replace("___", "_").replace(" ", "_")
    return key, folder.replace("___", " ").replace("_", " ")


def display_name(key: str) -> str:
    for k, name in TOMATO_FOLDER_MAP.values():
        if k == key:
            return name
    return key.replace("_", " ").title()
