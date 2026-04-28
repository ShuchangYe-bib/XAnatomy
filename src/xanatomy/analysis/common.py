"""
Shared utilities for disease-specific analyses.
"""

from __future__ import annotations

import re
import shutil
from collections import Counter
from pathlib import Path

import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

_POS_RE = re.compile(r"positive\s*\(([^)]+)\)", re.IGNORECASE)


def format_counts(label_order: list[str], counter: Counter, indent: str = "  ") -> str:
    extras = sorted(set(counter) - set(label_order))
    lines: list[str] = []
    for key in list(label_order) + extras:
        value = counter.get(key, 0)
        if value:
            lines.append(f"{indent}{key:<12s}= {value:5d}")
    return "\n".join(lines) if lines else f"{indent}(none)"


def parse_gt_severity(status_path: Path, disease: str) -> str | None:
    if not status_path.is_file():
        return None
    normalized_disease = disease.lower()
    try:
        text = status_path.read_text(encoding="utf-8")
    except OSError:
        return None
    for raw_line in text.splitlines():
        line = raw_line.strip().lower()
        if not line.startswith(f"{normalized_disease}:"):
            continue
        if "negative" in line:
            return "negative"
        match = _POS_RE.search(line)
        if match:
            return match.group(1).strip()
    return None


def is_acceptable(
    value: float | None,
    gt: str | None,
    bands: dict[str, tuple[float, float]],
    tolerate: float = 0.0,
) -> bool:
    if gt is None:
        return True
    if value is None:
        return False
    if gt == "unspecified":
        negative_upper = bands.get("negative", (float("-inf"), 0.0))[1]
        return value > negative_upper - tolerate
    if gt not in bands:
        return True
    lower, upper = bands[gt]
    return (lower - tolerate) <= value <= (upper + tolerate)


def largest_component(mask_2d: np.ndarray) -> np.ndarray:
    if not mask_2d.any():
        return mask_2d
    try:
        from scipy.ndimage import label as cc_label

        labeled, count = cc_label(mask_2d.astype(np.uint8))
    except ImportError:
        try:
            import cv2

            count, labeled = cv2.connectedComponents(mask_2d.astype(np.uint8))
            count -= 1
        except ImportError:
            return mask_2d
    if count <= 1:
        return mask_2d
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    return labeled == counts.argmax()


def centroid(mask_2d: np.ndarray) -> tuple[float, float] | None:
    cleaned = largest_component(mask_2d)
    if not cleaned.any():
        return None
    ys, xs = np.where(cleaned)
    return float(xs.mean()), float(ys.mean())


def list_image_files(input_dir: Path) -> list[Path]:
    return [path for path in input_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTS]


def copy_case_siblings(src_dir: Path, dst_dir: Path, skip_names: set[str]) -> None:
    for sibling in src_dir.iterdir():
        if not sibling.is_file() or sibling.name in skip_names:
            continue
        shutil.copy2(sibling, dst_dir / sibling.name)
