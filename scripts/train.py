"""Train the tomato disease classifier (CPU-friendly) and publish it to models/production.

Usage:
  python scripts/prepare_dataset.py
  python scripts/train.py --head-epochs 2 --finetune-epochs 4 --train-cap 400
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import get_settings
from src.ml.train import train


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--head-epochs", type=int, default=2)
    ap.add_argument("--finetune-epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--train-cap", type=int, default=400, help="max training images per class (0 = no cap)")
    ap.add_argument("--val-cap", type=int, default=150, help="max validation images per class (0 = no cap)")
    ap.add_argument("--time-budget-min", type=float, default=None)
    ap.add_argument("--resume", default=None, help="checkpoint to continue from (skips the head stage)")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    meta = train(get_settings(), a.head_epochs, a.finetune_epochs, a.batch_size, a.train_cap or None,
                 a.val_cap or None, seed=a.seed, time_budget_min=a.time_budget_min, resume_from=a.resume,
                 log=lambda m: print(m, flush=True))
    print("Done. Model version:", meta["model_version"])


if __name__ == "__main__":
    main()
