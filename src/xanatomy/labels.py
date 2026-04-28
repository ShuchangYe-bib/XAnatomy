"""
Label utilities and stable color generation.
"""

from __future__ import annotations

import colorsys
import json
from functools import lru_cache
from pathlib import Path

RESOURCE_DIR = Path(__file__).resolve().parent / "resources"
LABEL_UNIVERSE_PATH = RESOURCE_DIR / "label_universe.json"

with LABEL_UNIVERSE_PATH.open(encoding="utf-8") as handle:
    label_universe: list[str] = json.load(handle)

num_classes = len(label_universe)
id2label: dict[int, str] = {index: label for index, label in enumerate(label_universe)}
label2id: dict[str, int] = {label: index for index, label in enumerate(label_universe)}


def _make_palette(n: int) -> list[tuple[int, int, int]]:
    try:
        import colorcet as cc  # type: ignore

        cmap = cc.cm.glasbey_bw_minc_20
        palette: list[tuple[int, int, int]] = []
        for index in range(n):
            red, green, blue, _ = cmap(index % cmap.N)
            palette.append((int(red * 255), int(green * 255), int(blue * 255)))
        return palette
    except Exception:
        pass

    sv_cycle = [
        (0.95, 0.90),
        (0.55, 0.95),
        (0.85, 0.70),
        (0.70, 0.55),
        (0.95, 0.50),
        (0.60, 0.85),
        (0.80, 0.95),
        (0.50, 0.70),
        (0.90, 0.80),
        (0.65, 0.60),
        (0.75, 0.90),
        (0.85, 0.55),
    ]
    palette = []
    for index in range(n):
        hue = (index * 0.618033988749895) % 1.0
        saturation, value = sv_cycle[index % len(sv_cycle)]
        red, green, blue = colorsys.hsv_to_rgb(hue, saturation, value)
        palette.append((int(red * 255), int(green * 255), int(blue * 255)))
    return palette


colors = _make_palette(num_classes)

_GROUP_KEYWORDS: dict[str, tuple[str, ...]] = {
    "lungs": ("lung",),
    "heart": ("heart", "atrium", "ventricle", "auricle"),
    "vasculatory": (
        "aorta",
        "vena",
        "vein",
        "artery",
        "trunk",
        "portal",
        "iliac",
        "carotid",
        "subclavian",
        "pulmonary",
        "brachiocephalic",
    ),
    "vessels": (
        "aorta",
        "vena",
        "vein",
        "artery",
        "trunk",
        "portal",
        "iliac",
        "carotid",
        "subclavian",
    ),
    "airways": ("trachea", "bronchie", "larynx"),
    "respiratory": ("lung", "trachea", "bronchie", "larynx", "pleural"),
    "mediastinum": ("esophagus", "thymus", "mediastinum"),
    "ribs": ("rib",),
    "spine": ("vertebrae", "spinal"),
    "bones": (
        "rib",
        "vertebrae",
        "scapula",
        "clavicle",
        "sternum",
        "humerus",
        "femur",
        "skull",
        "sacrum",
        "hip",
        "bone",
        "costal cartilage",
    ),
    "abdominal": (
        "liver",
        "kidney",
        "spleen",
        "stomach",
        "pancreas",
        "intestine",
        "colon",
        "duodenum",
        "gallbladder",
        "adrenal",
        "abdominal",
        "bowel",
        "bladder",
    ),
    "abdomen": (
        "liver",
        "kidney",
        "spleen",
        "stomach",
        "pancreas",
        "intestine",
        "colon",
        "duodenum",
        "gallbladder",
        "adrenal",
        "abdominal",
    ),
    "thyroid": ("thyroid",),
    "breast": ("breast",),
}


@lru_cache(maxsize=None)
def group_ids(name: str) -> tuple[int, ...]:
    """
    Resolve a named coarse anatomical group to class ids.
    """
    normalized = name.lower()
    if normalized not in _GROUP_KEYWORDS:
        raise KeyError(f"unknown group {name!r}. options: {sorted(_GROUP_KEYWORDS)}")
    keywords = _GROUP_KEYWORDS[normalized]
    return tuple(
        index
        for index, label in enumerate(label_universe)
        if any(keyword in label for keyword in keywords)
    )


def resolve(names: str | int | list[str] | list[int]) -> list[int]:
    """
    Accept exact labels, substring queries, group names, or integer ids.
    """
    if isinstance(names, (str, int)):
        names = [names]

    resolved: set[int] = set()
    for item in names:
        if isinstance(item, int):
            resolved.add(item)
            continue
        if item in label2id:
            resolved.add(label2id[item])
            continue
        lowered = item.lower()
        if lowered in _GROUP_KEYWORDS:
            resolved.update(group_ids(lowered))
            continue
        matches = [index for index, label in enumerate(label_universe) if lowered in label.lower()]
        if not matches:
            raise KeyError(f"no label or group matches {item!r}")
        resolved.update(matches)
    return sorted(resolved)


def list_groups() -> list[str]:
    return sorted(_GROUP_KEYWORDS)
