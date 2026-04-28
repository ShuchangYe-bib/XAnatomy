"""
High-level public API for XAnatomy inference.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

from xanatomy.io import load_image, resize_mask, save_mask
from xanatomy.labels import num_classes
from xanatomy.models import build_segformer

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = _REPO_ROOT / "artifacts" / "checkpoints" / "segformer_b3_best.pt"
DEFAULT_CONFIG = Path(__file__).resolve().parent / "resources" / "segformer_config.yaml"


class XAnatomy(nn.Module):
    """
    Anatomy segmentation for 2D chest projections.

    Parameters
    ----------
    checkpoint:
        Optional checkpoint path. If omitted, uses `XANATOMY_CHECKPOINT` when
        set, otherwise `artifacts/checkpoints/segformer_b3_best.pt`.
    config:
        YAML config path. Defaults to the bundled package resource.
    device:
        `"cuda"`, `"cpu"`, or `"auto"`.
    threshold:
        Sigmoid threshold used for binary mask generation.
    """

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        config: str | Path | None = None,
        device: str = "auto",
        threshold: float = 0.5,
    ) -> None:
        super().__init__()

        checkpoint_env = os.environ.get("XANATOMY_CHECKPOINT")
        ckpt_path = Path(checkpoint or checkpoint_env or DEFAULT_CHECKPOINT)
        cfg_path = Path(config) if config else DEFAULT_CONFIG

        if not ckpt_path.is_file():
            raise FileNotFoundError(
                "checkpoint not found. Expected "
                f"{ckpt_path}. Either place the weight file there, pass "
                "`checkpoint=...`, or set `XANATOMY_CHECKPOINT`."
            )
        if not cfg_path.is_file():
            raise FileNotFoundError(f"config not found: {cfg_path}")

        with cfg_path.open(encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle) or {}

        model_cfg = cfg.get("model", {})
        train_cfg = cfg.get("train", {})
        self.variant = str(model_cfg.get("encoder", "b3"))

        image_size = train_cfg.get("image_size", [256, 256])
        if isinstance(image_size, (list, tuple)):
            self.image_size = (int(image_size[0]), int(image_size[1]))
        else:
            self.image_size = (int(image_size), int(image_size))

        self.threshold = float(threshold)
        self.num_classes = num_classes

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.model = build_segformer(num_classes=self.num_classes, variant=self.variant)
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = state.get("model_state_dict", state) if isinstance(state, dict) else state
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device).eval()

    @torch.no_grad()
    def predict(
        self,
        source,
        return_probs: bool = False,
        output_size: tuple[int, int] | None = None,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Segment a single image.

        Returns a `[C, H, W]` boolean mask. If `return_probs=True`, also returns
        the resized float probability map.
        """
        loaded = load_image(source, image_size=self.image_size)
        logits = self.model(loaded["tensor"].to(self.device))
        probs = torch.sigmoid(logits.float())[0]
        mask = probs > self.threshold

        if output_size is None:
            output_size = loaded["original"].shape[:2]
        mask_np = resize_mask(mask, size=output_size)

        if not return_probs:
            return mask_np

        import torch.nn.functional as F

        probs_resized = F.interpolate(
            probs.unsqueeze(0),
            size=output_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).cpu().numpy().astype(np.float32)
        return mask_np, probs_resized

    @torch.no_grad()
    def predict_folder(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
        fmt: str = "npz",
        extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg"),
    ) -> list[Path]:
        """
        Segment every matching image in `input_dir` and write masks to `output_dir`.
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        written: list[Path] = []
        for image_path in sorted(input_dir.iterdir()):
            if image_path.suffix.lower() not in extensions:
                continue
            mask = self.predict(image_path)
            out_path = output_dir / f"{image_path.stem}.{fmt}"
            save_mask(mask, out_path, fmt=fmt)
            written.append(out_path)
        return written

    def __repr__(self) -> str:
        return (
            f"XAnatomy(variant=segformer-{self.variant}, "
            f"num_classes={self.num_classes}, image_size={self.image_size}, "
            f"device={self.device})"
        )
