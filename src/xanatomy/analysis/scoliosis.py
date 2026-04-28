"""
Scoliosis analysis derived from segmented vertebrae centroids.
"""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFont

from xanatomy import XAnatomy
from xanatomy.analysis.common import (
    centroid,
    copy_case_siblings,
    format_counts,
    is_acceptable,
    list_image_files,
    parse_gt_severity,
)
from xanatomy.labels import label2id
from xanatomy.visualize import visualize_labels

DISEASE = "scoliosis"
FILTER_BANDS: dict[str, tuple[float, float]] = {
    "negative": (float("-inf"), 10.0),
    "mild": (10.0, 25.0),
    "moderate": (25.0, 45.0),
    "severe": (45.0, float("inf")),
}
PRED_ORDER = ["negative", "mild", "moderate", "severe", "unknown"]
GT_ORDER = ["negative", "mild", "moderate", "severe", "unspecified", "no-gt"]
GENERATED_NAMES = {"segmentation.png", "interpretation.png", "diagnosis.txt"}
_CANDIDATE_VERTEBRAE = [f"thoracic vertebrae {i} (t{i})" for i in range(1, 13)] + ["lumbar vertebrae 1 (l1)"]
SPINE_LABELS = [label for label in _CANDIDATE_VERTEBRAE if label in label2id]
SPINE_IDS = [label2id[label] for label in SPINE_LABELS]
SEG_LABELS = list(SPINE_LABELS)
GREEN = (254, 44, 85)
BLACK = (0, 0, 0)
DOT_COLOR = (37, 244, 238)


def parse_color(value: str) -> tuple[int, int, int]:
    return tuple(ImageColor.getrgb(value))[:3]


def spine_centroids(mask: np.ndarray) -> list[tuple[float, float]]:
    points = [centroid(mask[class_id]) for class_id in SPINE_IDS]
    filtered = [point for point in points if point is not None]
    filtered.sort(key=lambda point: point[1])
    return filtered


def has_centroids_both_sides(points: list[tuple[float, float]], eps: float = 1.0) -> bool:
    if len(points) < 3:
        return False
    top = points[0]
    bottom = points[-1]
    line_dx = bottom[0] - top[0]
    line_dy = bottom[1] - top[1]
    has_positive = False
    has_negative = False
    for x, y in points[1:-1]:
        cross = line_dx * (top[1] - y) - (top[0] - x) * line_dy
        if cross > eps:
            has_positive = True
        elif cross < -eps:
            has_negative = True
        if has_positive and has_negative:
            return True
    return False


def cobb_angle_degrees(points: list[tuple[float, float]]) -> tuple[float, int] | None:
    if len(points) < 3:
        return None
    top = points[0]
    bottom = points[-1]
    line_dx = bottom[0] - top[0]
    line_dy = bottom[1] - top[1]
    line_length = math.hypot(line_dx, line_dy)
    if line_length == 0:
        return None

    best_distance = -1.0
    best_index = 0
    for index in range(1, len(points) - 1):
        x, y = points[index]
        distance = abs(line_dx * (top[1] - y) - (top[0] - x) * line_dy) / line_length
        if distance > best_distance:
            best_distance = distance
            best_index = index

    apex = points[best_index]
    v1_x, v1_y = apex[0] - top[0], apex[1] - top[1]
    v2_x, v2_y = bottom[0] - apex[0], bottom[1] - apex[1]
    n1 = math.hypot(v1_x, v1_y)
    n2 = math.hypot(v2_x, v2_y)
    if n1 == 0 or n2 == 0:
        return None
    cosine = (v1_x * v2_x + v1_y * v2_y) / (n1 * n2)
    cosine = max(-1.0, min(1.0, cosine))
    return math.degrees(math.acos(cosine)), best_index


def severity_from_scd(scd_deg: float | None) -> str:
    if scd_deg is None:
        return "unknown"
    if scd_deg < 10:
        return "negative"
    if scd_deg < 25:
        return "mild"
    if scd_deg < 45:
        return "moderate"
    return "severe"


def render_segmentation(rgb: Image.Image, mask: np.ndarray, alpha: float = 0.35) -> Image.Image:
    return visualize_labels(rgb, mask, SEG_LABELS, alpha=alpha)


def render_interpretation(
    rgb: Image.Image,
    mask: np.ndarray,
    *,
    points: list[tuple[float, float]] | None = None,
    line_width: int = 4,
    point_radius: int = 5,
    font_size: int | None = None,
    reference_color: tuple[int, int, int] = GREEN,
    connector_color: tuple[int, int, int] = BLACK,
    point_color: tuple[int, int, int] = DOT_COLOR,
) -> tuple[Image.Image, float | None, int]:
    height, width = mask.shape[1:]
    image = rgb.convert("RGB").resize((width, height), Image.BILINEAR).copy()
    draw = ImageDraw.Draw(image)

    if points is None:
        points = spine_centroids(mask)

    scd = None
    if len(points) >= 2:
        top = points[0]
        bottom = points[-1]
        draw.line([top, bottom], fill=reference_color, width=line_width)

        dy_ref = bottom[1] - top[1]

        def x_on_chord(y: float) -> float:
            if dy_ref == 0:
                return top[0]
            return top[0] + (bottom[0] - top[0]) * (y - top[1]) / dy_ref

        connector_width = max(1, line_width - 2)
        for x, y in points:
            draw.line([(x_on_chord(y), y), (x, y)], fill=connector_color, width=connector_width)

        draw.line([tuple(point) for point in points], fill=point_color, width=line_width)
        for x, y in points:
            draw.ellipse(
                [(x - point_radius, y - point_radius), (x + point_radius, y + point_radius)],
                fill=point_color,
                outline=(0, 0, 0),
                width=1,
            )

        result = cobb_angle_degrees(points)
        if result is not None:
            scd, apex_index = result
            apex = points[apex_index]
            draw.line([(x_on_chord(apex[1]), apex[1]), (apex[0], apex[1])], fill=connector_color, width=line_width)
            draw.ellipse(
                [
                    (apex[0] - point_radius - 2, apex[1] - point_radius - 2),
                    (apex[0] + point_radius + 2, apex[1] + point_radius + 2),
                ],
                outline=connector_color,
                width=2,
            )

    if scd is not None:
        size = font_size if font_size is not None else max(12, width // 22)
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", size)
        except Exception:
            font = ImageFont.load_default()
        text = f"SCD={scd:.3f}°  {severity_from_scd(scd).upper()}"
        pad = 4
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.rectangle([(pad, pad), (pad + text_width + 2 * pad, pad + text_height + 2 * pad)], fill=(0, 0, 0))
        draw.text((pad * 2, pad * 2), text, fill=(255, 255, 255), font=font)

    return image, scd, len(points)


def analyse_image(
    model: XAnatomy,
    image_path: Path,
    *,
    threshold: float,
    line_width: int,
    point_radius: int,
    reference_color: tuple[int, int, int] = GREEN,
    connector_color: tuple[int, int, int] = BLACK,
    point_color: tuple[int, int, int] = DOT_COLOR,
) -> dict:
    rgb = Image.open(image_path).convert("RGB")
    _, probs = model.predict(rgb, return_probs=True)
    mask = probs > threshold
    points = spine_centroids(mask)
    segmentation = render_segmentation(rgb, mask)
    interpretation, scd, vertebrae_used = render_interpretation(
        rgb,
        mask,
        points=points,
        line_width=line_width,
        point_radius=point_radius,
        reference_color=reference_color,
        connector_color=connector_color,
        point_color=point_color,
    )
    return {
        "image_path": image_path,
        "segmentation": segmentation,
        "interpretation": interpretation,
        "scd": scd,
        "severity": severity_from_scd(scd),
        "vertebrae_used": vertebrae_used,
        "centroids": points,
    }


def write_diagnosis(out_path: Path, image_path: Path, scd: float | None, severity: str, n_used: int) -> None:
    scd_text = f"{scd:.3f} deg" if scd is not None else "unavailable"
    out_path.write_text(
        f"image: {image_path.name}\n"
        f"severity: {severity}\n"
        f"SCD: {scd_text}\n"
        f"vertebrae_detected: {n_used}\n"
        "reference: Mild 10-25 deg, Moderate 25-45 deg, Severe >=45 deg\n",
        encoding="utf-8",
    )


def show_single(result: dict) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 6))
    axes[0].imshow(result["segmentation"])
    axes[0].set_title("Segmentation (spine)", fontsize=12, fontweight="bold")
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    title = "Interpretation"
    if result["scd"] is not None:
        title += f"   SCD={result['scd']:.2f}°  ({result['severity']})"
    axes[1].imshow(result["interpretation"])
    axes[1].set_title(title, fontsize=12, fontweight="bold")
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    fig.suptitle(result["image_path"].name, fontsize=13)
    fig.tight_layout()
    plt.show()


def process_folder(
    model: XAnatomy,
    input_dir: Path,
    output_dir: Path,
    *,
    threshold: float,
    line_width: int,
    point_radius: int,
    filter_: bool = False,
    tolerate: float = 0.0,
    min_vertebrae: int = 10,
    require_both_sides: bool = False,
    keep_unspecified: bool = False,
    reference_color: tuple[int, int, int] = GREEN,
    connector_color: tuple[int, int, int] = BLACK,
    point_color: tuple[int, int, int] = DOT_COLOR,
) -> int:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    images = list_image_files(input_dir)
    if not images:
        print(f"[scoliosis] no images found under {input_dir}")
        return 0

    written = 0
    skipped = 0
    insufficient = 0
    unilateral = 0
    unspecified_skipped = 0
    pred_counts: Counter = Counter()
    kept_by_gt: Counter = Counter()
    skipped_by_gt: Counter = Counter()

    for index, image_path in enumerate(images, 1):
        relative_parent = image_path.relative_to(input_dir).parent
        try:
            result = analyse_image(
                model,
                image_path,
                threshold=threshold,
                line_width=line_width,
                point_radius=point_radius,
                reference_color=reference_color,
                connector_color=connector_color,
                point_color=point_color,
            )
        except Exception as exc:
            print(f"  [{index}/{len(images)}] {image_path.relative_to(input_dir)} - FAILED: {exc}")
            continue

        scd_str = f"{result['scd']:.2f}°" if result["scd"] is not None else "n/a"
        pred_counts[result["severity"]] += 1
        gt = parse_gt_severity(image_path.parent / "status.txt", DISEASE)
        gt_label = gt if gt is not None else "no-gt"

        if not keep_unspecified and gt in (None, "unspecified"):
            unspecified_skipped += 1
            continue
        if min_vertebrae > 0 and result["vertebrae_used"] < min_vertebrae:
            insufficient += 1
            continue
        if require_both_sides and not has_centroids_both_sides(result["centroids"]):
            unilateral += 1
            continue
        if filter_ and not is_acceptable(result["scd"], gt, FILTER_BANDS, tolerate):
            skipped += 1
            skipped_by_gt[gt_label] += 1
            continue
        if filter_:
            kept_by_gt[gt_label] += 1

        case_dir = output_dir / relative_parent
        case_dir.mkdir(parents=True, exist_ok=True)
        copy_case_siblings(image_path.parent, case_dir, GENERATED_NAMES)
        result["segmentation"].save(case_dir / "segmentation.png")
        result["interpretation"].save(case_dir / "interpretation.png")
        write_diagnosis(case_dir / "diagnosis.txt", image_path, result["scd"], result["severity"], result["vertebrae_used"])
        written += 1
        print(
            f"  [{index}/{len(images)}] {image_path.relative_to(input_dir)} -> "
            f"SCD={scd_str} ({result['severity']}, {result['vertebrae_used']} vertebrae)"
        )

    print()
    print(f"[scoliosis] predicted severity counts (all {sum(pred_counts.values())} processed):")
    print(format_counts(PRED_ORDER, pred_counts))
    if filter_:
        print(f"[scoliosis] kept by GT severity ({sum(kept_by_gt.values())} cases):")
        print(format_counts(GT_ORDER, kept_by_gt))
        print(f"[scoliosis] skipped by GT severity ({sum(skipped_by_gt.values())} cases):")
        print(format_counts(GT_ORDER, skipped_by_gt))
    if insufficient:
        print(f"[scoliosis] skipped for insufficient vertebrae: {insufficient}")
    if unilateral:
        print(f"[scoliosis] skipped for unilateral centroids: {unilateral}")
    if unspecified_skipped:
        print(f"[scoliosis] skipped unspecified/no-gt: {unspecified_skipped}")
    return written
