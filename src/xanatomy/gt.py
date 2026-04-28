"""
Ground-truth mask helpers.
"""

from __future__ import annotations

import numpy as np

from xanatomy.labels import label2id, num_classes


def gt_mask_from_row(row: dict, view: str = "pa") -> np.ndarray:
    """
    Reassemble per-label PIL masks from a dataset row into `[C, H, W]`.
    """
    normalized_view = view.lower()
    masks = row.get(f"masks_{normalized_view}") or []
    labels = row.get(f"mask_labels_{normalized_view}") or []
    if not masks:
        raise ValueError(f"row has no masks_{normalized_view} entry")

    first = np.array(masks[0])
    height, width = first.shape[:2]
    output = np.zeros((num_classes, height, width), dtype=bool)
    for mask, label in zip(masks, labels):
        if label not in label2id:
            continue
        array = np.array(mask)
        if array.ndim == 3:
            array = array.max(axis=2)
        output[label2id[label]] |= array > 0
    return output
