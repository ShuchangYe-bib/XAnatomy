"""
2D visualization helpers for anatomy masks.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from xanatomy.io import to_rgb_array
from xanatomy.labels import colors, label_universe, resolve


def _as_image_rgb(image, target_size: tuple[int, int] | None = None) -> np.ndarray:
    array = to_rgb_array(image)
    if target_size is not None and array.shape[:2] != tuple(target_size):
        array = np.array(
            Image.fromarray(array).resize((target_size[1], target_size[0]), Image.BILINEAR)
        )
    return array


def overlay_mask(
    image,
    mask: np.ndarray,
    class_ids: list[int] | None = None,
    alpha: float = 0.45,
    draw_contours: bool = True,
) -> Image.Image:
    """
    Overlay a multi-label mask on an RGB image.
    """
    if mask.ndim != 3:
        raise ValueError(f"mask must be [C, H, W], got shape {mask.shape}")
    channels, height, width = mask.shape
    image_rgb = _as_image_rgb(image, target_size=(height, width)).astype(np.float32)

    if class_ids is None:
        class_ids = [channel for channel in range(channels) if mask[channel].any()]

    fill = np.zeros_like(image_rgb)
    for channel in class_ids:
        if channel >= channels or not mask[channel].any():
            continue
        fill[mask[channel].astype(bool)] = np.array(colors[channel], dtype=np.float32)

    blended = ((1.0 - alpha) * image_rgb + alpha * fill).clip(0, 255).astype(np.uint8)

    if draw_contours:
        try:
            import cv2

            for channel in class_ids:
                if channel >= channels or not mask[channel].any():
                    continue
                binary = mask[channel].astype(np.uint8) * 255
                contours, _ = cv2.findContours(
                    binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(blended, contours, -1, tuple(int(x) for x in colors[channel]), 1)
        except ImportError:
            pass

    return Image.fromarray(blended)


def visualize_labels(
    image,
    mask: np.ndarray,
    labels: str | list[str] | list[int],
    alpha: float = 0.45,
    draw_contours: bool = True,
    side_by_side: bool = False,
) -> Image.Image:
    """
    Resolve labels or groups and render an overlay.
    """
    class_ids = resolve(labels)
    overlay = overlay_mask(
        image,
        mask,
        class_ids=class_ids,
        alpha=alpha,
        draw_contours=draw_contours,
    )
    if not side_by_side:
        return overlay

    height, width = mask.shape[1:]
    image_rgb = _as_image_rgb(image, target_size=(height, width))
    combined = np.concatenate([image_rgb, np.array(overlay)], axis=1)
    return Image.fromarray(combined)


def legend(class_ids: list[int] | None = None, swatch: int = 18) -> Image.Image:
    """
    Build a color legend image.
    """
    if class_ids is None:
        class_ids = list(range(len(label_universe)))

    from PIL import ImageDraw, ImageFont

    rows = len(class_ids)
    row_height = swatch + 4
    width = 420
    image = Image.new("RGB", (width, max(rows * row_height, swatch)), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for row, class_id in enumerate(class_ids):
        y = row * row_height + 2
        draw.rectangle([(4, y), (4 + swatch, y + swatch)], fill=colors[class_id])
        draw.text(
            (4 + swatch + 6, y + 2),
            label_universe[class_id],
            fill=(20, 20, 20),
            font=font,
        )
    return image


def save_overlay(image: Image.Image, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
