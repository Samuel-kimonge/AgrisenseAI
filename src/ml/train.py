"""Two-stage transfer-learning training, calibration and model packaging."""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def train(settings, epochs_head: int = 2, epochs_finetune: int = 4, batch_size: int = 32,
          train_cap: int | None = 400, val_cap: int | None = 150, lr_head: float = 2e-3, lr_ft: float = 4e-4,
          seed: int = 42, time_budget_min: float | None = None, resume_from=None, log=print) -> dict:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    import timm

    from . import evaluate as ev
    from .classes import display_name
    from .dataset import LeafDataset, cap_split, load_manifest
    from .model import build_model, save_checkpoint, set_backbone_trainable, write_json

    t0 = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(max(1, settings.torch_threads))

    manifest = load_manifest(settings.processed_dir)
    class_keys = sorted(manifest["label"].unique())
    root, size = settings.raw_data_dir, settings.image_size
    tr_rows = cap_split(manifest, "train", train_cap, seed)
    va_rows = cap_split(manifest, "val", val_cap, seed)
    te_rows = manifest[manifest["split"] == "test"]
    log(f"classes={len(class_keys)} train={len(tr_rows)} val={len(va_rows)} test={len(te_rows)} size={size}")

    train_ds = LeafDataset(root, tr_rows, class_keys, size, train=True)
    val_ds = LeafDataset(root, va_rows, class_keys, size, train=False)
    log(f"datasets cached in {time.time()-t0:.0f}s")
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

    model = build_model(len(class_keys), settings.pretrained_weights)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    history: list[dict] = []
    best_acc, best_path = -1.0, settings.checkpoints_dir / "best.pt"
    resumed_epoch = 0
    if resume_from:
        from .model import load_checkpoint as _lc
        st, ck = _lc(resume_from)
        assert ck["class_keys"] == class_keys, "checkpoint classes differ from dataset"
        model.load_state_dict(st)
        resumed_epoch, best_acc = int(ck["epoch"]), float(ck["val_acc"])
        epochs_head = 0
        log(f"Resuming from {resume_from} (epoch {resumed_epoch}, val_acc {best_acc:.4f}); head stage skipped")
    stages = [("head", epochs_head, lr_head, False), ("finetune", epochs_finetune, lr_ft, True)]
    total_epochs = epochs_head + epochs_finetune
    ep_global = resumed_epoch
    stop = False

    for stage, n_ep, lr, unfreeze in stages:
        if n_ep <= 0:
            continue
        set_backbone_trainable(model, unfreeze)
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=n_ep * len(train_dl), pct_start=0.2)
        for ep in range(n_ep):
            ep_global += 1
            model.train()
            run_loss, seen, correct = 0.0, 0, 0
            for x, y in train_dl:
                opt.zero_grad()
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                opt.step()
                sched.step()
                run_loss += loss.item() * len(y)
                seen += len(y)
                correct += (out.argmax(1) == y).sum().item()
            vl, vy = ev.collect_logits(model, val_dl)
            vacc = float((vl.argmax(1) == vy).mean())
            rec = {"epoch": ep_global, "stage": stage, "train_loss": run_loss / seen, "train_acc": correct / seen,
                   "val_acc": vacc, "elapsed_s": round(time.time() - t0)}
            history.append(rec)
            log(json.dumps(rec))
            if vacc >= best_acc:
                best_acc = vacc
                save_checkpoint(best_path, model, {"class_keys": class_keys, "epoch": ep_global, "val_acc": vacc})
            if time_budget_min and (time.time() - t0) / 60 > time_budget_min:
                log("Time budget reached; stopping early.")
                stop = True
                break
        if stop:
            break

    # ---- restore best, calibrate, set thresholds ---------------------------
    from .model import load_checkpoint
    state, _ = load_checkpoint(best_path)
    model.load_state_dict(state)
    val_logits, val_labels = ev.collect_logits(model, val_dl)
    T = ev.fit_temperature(val_logits, val_labels)
    thr = ev.select_thresholds(val_logits, val_labels, T)
    log(f"temperature={T:.3f} thresholds={thr}")

    # ---- test on held-out images (lazy loading to save RAM) ------------------
    test_ds = LeafDataset(root, te_rows, class_keys, size, train=False, cache=False)
    test_dl = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)
    te_logits, te_labels = ev.collect_logits(model, test_dl)
    test_report = ev.full_report(te_logits, te_labels, class_keys, T, thr)
    val_report = ev.full_report(val_logits, val_labels, class_keys, T, thr)
    log(f"TEST accuracy={test_report['accuracy']:.4f} macroF1={test_report['macro_f1']:.4f} "
        f"ECE {test_report['ece_uncalibrated']:.4f}->{test_report['ece_calibrated']:.4f} "
        f"coverage={test_report['coverage_at_thresholds']:.3f} selective_acc={test_report['selective_accuracy']}")

    # ---- out-of-distribution check on leaves of OTHER crops ------------------
    ood_report = None
    ood_root = settings.raw_data_dir.parent / "ood_leaves"
    if ood_root.is_dir():
        rows = []
        rng = np.random.default_rng(seed)
        for d in sorted(p for p in ood_root.iterdir() if p.is_dir()):
            files = sorted(f for f in d.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"})
            pick = rng.choice(len(files), size=min(100, len(files)), replace=False)
            rows += [{"path": str(files[i].relative_to(ood_root)), "label": class_keys[0]} for i in pick]
        ood_ds = LeafDataset(ood_root, pd.DataFrame(rows), class_keys, size, train=False, cache=False)
        o_logits, _ = ev.collect_logits(model, DataLoader(ood_ds, batch_size=64))
        ood_report = ev.ood_abstention_rate(o_logits, T, thr)
        ood_report["note"] = "Healthy leaves of apple, maize, grape, bell pepper, potato and strawberry (never seen in training)."
        log(f"OOD other-crop leaves: {ood_report}")

    # ---- package ------------------------------------------------------------
    import torchvision
    meta = {
        "model_version": datetime.now(timezone.utc).strftime("tomato-mnv3l-%Y%m%d-%H%M"),
        "architecture": "MobileNetV3-Large (timm mobilenetv3_large_100), ImageNet-pretrained, fine-tuned",
        "class_keys": class_keys,
        "class_names": {k: display_name(k) for k in class_keys},
        "image_size": size,
        "temperature": T,
        "thresholds": thr,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "training": {"epochs_head": epochs_head, "epochs_finetune": epochs_finetune, "epochs_run": ep_global,
                     "batch_size": batch_size, "train_cap_per_class": train_cap, "val_cap_per_class": val_cap,
                     "n_train": int(len(tr_rows)), "n_val": int(len(va_rows)), "n_test": int(len(te_rows)),
                     "resumed_from_epoch": resumed_epoch or None, "seed": seed, "lr_head": lr_head, "lr_finetune": lr_ft,
                     "wall_clock_minutes": round((time.time() - t0) / 60, 1)},
        "metrics": {"test_accuracy": test_report["accuracy"], "test_macro_f1": test_report["macro_f1"],
                    "test_ece_calibrated": test_report["ece_calibrated"],
                    "test_coverage": test_report["coverage_at_thresholds"],
                    "test_selective_accuracy": test_report["selective_accuracy"],
                    "ood_other_crops": ood_report},
        "libraries": {"torch": torch.__version__, "torchvision": torchvision.__version__, "timm": timm.__version__},
        "dataset": "PlantVillage (raw/color), tomato subset - controlled-environment photos; see DATASET.md",
    }
    prod = settings.production_model_dir
    prod.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), prod / "model.pt")
    write_json(prod / "metadata.json", meta)
    ev_dir = settings.reports_dir / "evaluation"
    write_json(ev_dir / "test_report.json", test_report)
    write_json(ev_dir / "val_report.json", val_report)
    write_json(ev_dir / "training_history.json", history)
    if ood_report:
        write_json(ev_dir / "ood_report.json", ood_report)
    pd.DataFrame(test_report["confusion_matrix"], index=class_keys, columns=class_keys).to_csv(ev_dir / "confusion_matrix.csv")
    pd.DataFrame(test_report["classification_report"]).T.to_csv(ev_dir / "classification_report.csv")
    return meta
