"""Discover, validate and split the tomato dataset.

Usage:  python scripts/prepare_dataset.py [--raw data/raw/plantvillage_tomato]
Writes: data/processed/manifest.csv, data/processed/validation_report.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import get_settings
from src.ml.dataset import discover_dataset, save_manifest, stratified_split, validate_dataset


def main() -> None:
    s = get_settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(s.raw_data_dir))
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    info = discover_dataset(Path(a.raw))
    print("Discovered classes:")
    for k in info.class_keys:
        print(f"  {k:32s} {info.counts[k]:5d}  ({info.folders[k]})")
    df, report = validate_dataset(info)
    df = stratified_split(df, seed=a.seed)
    report["split_counts"] = df.groupby(["split", "label"]).size().unstack(0).fillna(0).astype(int).to_dict()
    report["seed"] = a.seed
    save_manifest(df, report, s.processed_dir)
    print(json.dumps({k: report[k] for k in ("total_found", "total_kept", "problems")}, indent=2))
    print(df["split"].value_counts().to_string())
    print(f"Saved manifest to {s.processed_dir}")


if __name__ == "__main__":
    main()
