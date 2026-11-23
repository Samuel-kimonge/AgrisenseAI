"""Model construction (MobileNetV3-Large transfer learning) and checkpoint IO."""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn

ARCH = "mobilenetv3_large_100"


def build_model(num_classes: int, pretrained_weights: Path | None = None, dropout: float = 0.2) -> nn.Module:
    """MobileNetV3-Large with a new classification head.

    Backbone weights are the ImageNet weights shipped with timm
    (mobilenetv3_large_100_ra). If `pretrained_weights` is None or missing the
    network is randomly initialised (used by unit tests and inference, where the
    trained checkpoint overwrites everything anyway).
    """
    import timm
    kwargs = dict(num_classes=num_classes, drop_rate=dropout)
    if pretrained_weights and Path(pretrained_weights).exists():
        model = timm.create_model(
            ARCH, pretrained=True,
            pretrained_cfg_overlay=dict(file=str(pretrained_weights)),
            **kwargs,
        )
    else:
        model = timm.create_model(ARCH, pretrained=False, **kwargs)
    return model


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    head_prefixes = ("classifier", "conv_head")
    for name, p in model.named_parameters():
        is_head = name.startswith(head_prefixes)
        p.requires_grad = True if is_head else trainable


def save_checkpoint(path: Path, model: nn.Module, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "meta": meta}, path)


def load_checkpoint(path: Path, map_location="cpu") -> tuple[dict, dict]:
    ck = torch.load(path, map_location=map_location, weights_only=False)
    return ck["state_dict"], ck["meta"]


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))
