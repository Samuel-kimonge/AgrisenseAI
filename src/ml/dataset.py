"""Dataset discovery, integrity validation, splitting and torch Datasets."""
from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from PIL import Image, UnidentifiedImageError

from .classes import key_for_folder

IMG_EXTS = {".jpg", ".jpeg", ".png"}


@dataclass
class DatasetInfo:
    root: Path
    class_keys: list[str]          # sorted, defines label index order
    folders: dict[str, str]        # key -> folder name
    counts: dict[str, int]         # key -> number of files found


def discover_dataset(root: Path) -> DatasetInfo:
    """Discover class folders and file counts. Raises if nothing usable is found."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {root}")
    folders: dict[str, str] = {}
    counts: dict[str, int] = {}
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        files = [f for f in d.iterdir() if f.suffix.lower() in IMG_EXTS]
        if not files:
            continue
        key, _ = key_for_folder(d.name)
        folders[key] = d.name
        counts[key] = len(files)
    if len(folders) < 2:
        raise ValueError(f"Need at least 2 class folders with images in {root}, found {len(folders)}")
    return DatasetInfo(root=root, class_keys=sorted(folders), folders=folders, counts=counts)


def validate_dataset(info: DatasetInfo, min_side: int = 64) -> tuple[pd.DataFrame, dict]:
    """Verify every image decodes, is large enough, and is not an exact duplicate.

    Returns (clean manifest DataFrame, validation report dict).
    """
    rows, seen = [], {}
    problems = Counter()
    bad_examples: list[str] = []
    for key in info.class_keys:
        folder = info.root / info.folders[key]
        for f in sorted(folder.iterdir()):
            if f.suffix.lower() not in IMG_EXTS:
                continue
            try:
                with Image.open(f) as im:
                    im.verify()
                with Image.open(f) as im:
                    w, h = im.size
                    mode = im.mode
            except (UnidentifiedImageError, OSError, SyntaxError):
                problems["corrupt"] += 1
                bad_examples.append(str(f))
                continue
            if min(w, h) < min_side:
                problems["too_small"] += 1
                continue
            digest = hashlib.md5(f.read_bytes()).hexdigest()
            if digest in seen:
                problems["exact_duplicate"] += 1
                if seen[digest] != key:
                    problems["cross_class_duplicate"] += 1
                continue
            seen[digest] = key
            rows.append({"path": str(f.relative_to(info.root)), "label": key, "width": w, "height": h, "mode": mode, "md5": digest})
    df = pd.DataFrame(rows)
    report = {
        "root": str(info.root),
        "classes": info.class_keys,
        "files_found_per_class": info.counts,
        "files_kept_per_class": df["label"].value_counts().sort_index().to_dict() if len(df) else {},
        "total_found": int(sum(info.counts.values())),
        "total_kept": int(len(df)),
        "problems": dict(problems),
        "corrupt_examples": bad_examples[:10],
    }
    return df, report


def stratified_split(df: pd.DataFrame, val_frac: float = 0.15, test_frac: float = 0.15, seed: int = 42) -> pd.DataFrame:
    """Stratified train/val/test split; adds a 'split' column."""
    rng = random.Random(seed)
    split = pd.Series(index=df.index, dtype=object)
    for label, idx in df.groupby("label").groups.items():
        idx = list(idx)
        rng.shuffle(idx)
        n = len(idx)
        n_test = max(1, int(round(n * test_frac)))
        n_val = max(1, int(round(n * val_frac)))
        split.loc[idx[:n_test]] = "test"
        split.loc[idx[n_test:n_test + n_val]] = "val"
        split.loc[idx[n_test + n_val:]] = "train"
    out = df.copy()
    out["split"] = split
    return out


def cap_split(manifest: pd.DataFrame, split: str, cap: int | None, seed: int = 42) -> pd.DataFrame:
    """Return only rows of `split`, with at most `cap` images per class.
    Used to keep CPU training/validation tractable on small machines."""
    rows = manifest[manifest["split"] == split]
    if not cap:
        return rows
    parts = [g.sample(n=min(cap, len(g)), random_state=seed) for _, g in rows.groupby("label")]
    return pd.concat(parts).sort_index()


def save_manifest(manifest: pd.DataFrame, report: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out_dir / "manifest.csv", index=False)
    (out_dir / "validation_report.json").write_text(json.dumps(report, indent=2))


def load_manifest(processed_dir: Path) -> pd.DataFrame:
    p = Path(processed_dir) / "manifest.csv"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found. Run: python scripts/prepare_dataset.py")
    return pd.read_csv(p)


# --------------------------- torch dataset -----------------------------------
def make_transforms(image_size: int, train: bool):
    import torchvision.transforms as T
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    if train:
        return T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.6, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(25),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.03),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
    return T.Compose([T.Resize((image_size, image_size)), T.ToTensor(), T.Normalize(mean, std)])


class LeafDataset:
    """Map-style dataset over manifest rows. Images are decoded once, resized to
    ~1.15x the target size and cached in RAM (small footprint at 3-4k images)."""

    def __init__(self, root: Path, rows: pd.DataFrame, class_keys: list[str], image_size: int, train: bool, cache: bool = True):
        self.root = Path(root)
        self.transform = make_transforms(image_size, train)
        self.class_to_idx = {k: i for i, k in enumerate(class_keys)}
        self.pre = int(image_size * 1.15) if train else image_size
        self.paths = [r.path for r in rows.itertuples()]
        self.labels = [self.class_to_idx[r.label] for r in rows.itertuples()]
        self.cache = cache
        self.images = [self._load(p) for p in self.paths] if cache else None

    def _load(self, rel: str) -> Image.Image:
        with Image.open(self.root / rel) as im:
            return im.convert("RGB").resize((self.pre, self.pre), Image.BILINEAR)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        img = self.images[i] if self.cache else self._load(self.paths[i])
        return self.transform(img), self.labels[i]
