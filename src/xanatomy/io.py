"""
Image loading and mask persistence helpers.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def to_rgb_array(source) -> np.ndarray:
    """
    Convert path, PIL image, or numpy array to `H x W x 3` uint8 RGB.
    """
    if isinstance(source, (str, os.PathLike, Path)):
        return np.array(Image.open(str(source)).convert("RGB"))
    if isinstance(source, Image.Image):
        return np.array(source.convert("RGB"))
    if isinstance(source, np.ndarray):
        array = source
        if array.ndim == 2:
            array = np.stack([array, array, array], axis=-1)
        if array.dtype != np.uint8:
            if array.max() <= 1.0:
                array = (array * 255).clip(0, 255).astype(np.uint8)
            else:
                array = array.clip(0, 255).astype(np.uint8)
        return array
    raise TypeError(f"unsupported image type: {type(source)!r}")


def load_image(source, image_size: int | tuple[int, int] = 256) -> dict:
    """
    Prepare an image for SegFormer inference.
    """
    rgb_full = to_rgb_array(source)
    if isinstance(image_size, int):
        image_size = (image_size, image_size)
    height, width = image_size

    resized = np.array(Image.fromarray(rgb_full).resize((width, height), Image.BILINEAR))
    tensor = torch.from_numpy(resized).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    tensor = (tensor - _MEAN) / _STD

    return {
        "tensor": tensor,
        "rgb": resized,
        "original": rgb_full,
        "size": (height, width),
    }


def save_mask(mask: np.ndarray, out_path: str | Path, fmt: str = "npz") -> None:
    """
    Persist a binary multi-label mask to disk.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = fmt.lower().lstrip(".")

    if fmt == "npz":
        np.savez_compressed(out_path, mask=mask.astype(np.uint8))
        return
    if fmt == "npy":
        np.save(out_path, mask.astype(np.uint8))
        return
    raise ValueError(f"unsupported mask format: {fmt}")


def resize_mask(mask: torch.Tensor, size: tuple[int, int]) -> np.ndarray:
    """
    Nearest-neighbor resize of a `[C, H, W]` mask.
    """
    if mask.ndim == 3:
        mask = mask.unsqueeze(0)
    output = F.interpolate(mask.float(), size=size, mode="nearest")
    return output.squeeze(0).bool().cpu().numpy()
