"""Evaluation, calibration and abstention-threshold selection."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, softmax
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def collect_logits(model, loader) -> tuple[np.ndarray, np.ndarray]:
    import torch
    model.eval()
    logits, labels = [], []
    with torch.inference_mode():
        for x, y in loader:
            logits.append(model(x).float().numpy())
            labels.append(np.asarray(y))
    return np.concatenate(logits), np.concatenate(labels)


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Temperature scaling: find T>0 minimising validation NLL."""
    def nll(logT: float) -> float:
        lp = logits / np.exp(logT)
        lp = lp - logsumexp(lp, axis=1, keepdims=True)
        return float(-lp[np.arange(len(labels)), labels].mean())
    res = minimize_scalar(nll, bounds=(-2, 3), method="bounded")
    return float(np.exp(res.x))


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    conf = probs.max(1)
    pred = probs.argmax(1)
    correct = (pred == labels).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def decision_features(logits: np.ndarray, temperature: float) -> dict[str, np.ndarray]:
    probs = softmax(logits / temperature, axis=1)
    part = np.sort(probs, axis=1)
    top, second = part[:, -1], part[:, -2]
    ent = -(probs * np.log(probs + 1e-12)).sum(1) / np.log(probs.shape[1])
    return {"probs": probs, "conf": top, "margin": top - second, "entropy": ent, "max_logit": logits.max(1)}


def select_thresholds(val_logits, val_labels, temperature, target_selective_acc=0.97, margin_thr=0.15,
                      logit_quantile=0.02) -> dict:
    """Choose abstention thresholds on the validation set only."""
    f = decision_features(val_logits, temperature)
    pred = f["probs"].argmax(1)
    correct = pred == val_labels
    chosen = 0.90
    for thr in np.arange(0.50, 0.995, 0.01):
        acc_mask = (f["conf"] >= thr) & (f["margin"] >= margin_thr)
        if acc_mask.sum() >= 0.5 * len(val_labels) and correct[acc_mask].mean() >= target_selective_acc:
            chosen = float(round(thr, 2))
            break
    return {
        "min_confidence": chosen,
        "min_margin": margin_thr,
        # raw max-logit floor: a low maximum logit means "nothing in the model matches"
        "min_max_logit": float(np.quantile(f["max_logit"][correct], logit_quantile)) if correct.any() else 0.0,
    }


def apply_abstention(feats: dict, thr: dict) -> np.ndarray:
    """Boolean array: True where the system would ABSTAIN."""
    return (feats["conf"] < thr["min_confidence"]) | (feats["margin"] < thr["min_margin"]) | (feats["max_logit"] < thr["min_max_logit"])


def full_report(logits, labels, class_keys, temperature, thresholds) -> dict:
    feats = decision_features(logits, temperature)
    pred = feats["probs"].argmax(1)
    abstain = apply_abstention(feats, thresholds)
    raw_probs = softmax(logits, axis=1)
    cr = classification_report(labels, pred, labels=list(range(len(class_keys))), target_names=class_keys,
                               output_dict=True, zero_division=0)
    cm = confusion_matrix(labels, pred, labels=list(range(len(class_keys))))
    accepted = ~abstain
    return {
        "n": int(len(labels)),
        "accuracy": float(accuracy_score(labels, pred)),
        "macro_f1": float(f1_score(labels, pred, average="macro")),
        "weighted_f1": float(f1_score(labels, pred, average="weighted")),
        "ece_uncalibrated": expected_calibration_error(raw_probs, labels),
        "ece_calibrated": expected_calibration_error(feats["probs"], labels),
        "coverage_at_thresholds": float(accepted.mean()),
        "selective_accuracy": float((pred[accepted] == labels[accepted]).mean()) if accepted.any() else None,
        "classification_report": cr,
        "confusion_matrix": cm.tolist(),
        "class_keys": class_keys,
    }


def ood_abstention_rate(logits, temperature, thresholds) -> dict:
    feats = decision_features(logits, temperature)
    ab = apply_abstention(feats, thresholds)
    return {"n": int(len(ab)), "abstained": int(ab.sum()), "abstention_rate": float(ab.mean()),
            "mean_confidence": float(feats["conf"].mean())}
