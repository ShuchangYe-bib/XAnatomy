"""
Kyphosis analysis derived from segmented vertebrae geometry.
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
    largest_component,
    list_image_files,
    parse_gt_severity,
)
from xanatomy.labels import label2id
from xanatomy.visualize import visualize_labels

DISEASE = "kyphosis"
FILTER_BANDS: dict[str, tuple[float, float]] = {
    "negative": (float("-inf"), 50.0),
    "mild": (50.0, 60.0),
    "moderate": (60.0, 70.0),
    "severe": (70.0, float("inf")),
}
PRED_ORDER = ["negative", "mild", "moderate", "severe", "unknown"]
GT_ORDER = ["negative", "mild", "moderate", "severe", "unspecified", "no-gt"]
GENERATED_NAMES = {"segmentation.png", "interpretation.png", "diagnosis.txt"}
_CANDIDATE_VERTEBRAE = [f"thoracic vertebrae {i} (t{i})" for i in range(1, 13)] + ["lumbar vertebrae 1 (l1)"]
SPINE_LABELS = [label for label in _CANDIDATE_VERTEBRAE if label in label2id]
SPINE_IDS = [label2id[label] for label in SPINE_LABELS]
SEG_LABELS = list(SPINE_LABELS)
TOP_ANCHOR_PREFERENCE = [
    "thoracic vertebrae 4 (t4)",
    "thoracic vertebrae 5 (t5)",
    "thoracic vertebrae 3 (t3)",
    "thoracic vertebrae 2 (t2)",
    "thoracic vertebrae 1 (t1)",
]
BOT_ANCHOR_PREFERENCE = [
    "thoracic vertebrae 12 (t12)",
    "thoracic vertebrae 11 (t11)",
    "lumbar vertebrae 1 (l1)",
    "thoracic vertebrae 10 (t10)",
]
ENDPLATE_COLOR = (254, 44, 85)
AXIS_COLOR = (0, 0, 0)
DOT_COLOR = (37, 244, 238)


def parse_color(value: str) -> tuple[int, int, int]:
    return tuple(ImageColor.getrgb(value))[:3]


def vertebra_axis(mask_2d: np.ndarray) -> tuple[float, float, float, float] | None:
    cleaned = largest_component(mask_2d)
    if not cleaned.any():
        return None
    ys, xs = np.where(cleaned)
    if len(xs) < 4:
        return None
    points = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
    center_x, center_y = points.mean(axis=0)
    centered = points - (center_x, center_y)
    covariance = np.cov(centered.T)
    if not np.isfinite(covariance).all():
        return None
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    long_axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    norm = math.hypot(float(long_axis[0]), float(long_axis[1]))
    if norm == 0:
        return None
    return float(center_x), float(center_y), float(long_axis[0] / norm), float(long_axis[1] / norm)


def pick_anchor(mask: np.ndarray, preferences: list[str]) -> tuple[str, tuple[float, float, float, float]] | None:
    for label in preferences:
        class_id = label2id.get(label)
        if class_id is None:
            continue
        axis = vertebra_axis(mask[class_id])
        if axis is not None:
            return label, axis
    return None


def endplate_endpoints(
    mask_2d: np.ndarray,
    axis: tuple[float, float, float, float],
    edge: str,
    length: float,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    cleaned = largest_component(mask_2d)
    if not cleaned.any():
        return None
    _, _, vx, vy = axis
    ys, xs = np.where(cleaned)
    if edge == "top":
        edge_y = ys.min()
        band = ys <= edge_y + 2
    elif edge == "bot":
        edge_y = ys.max()
        band = ys >= edge_y - 2
    else:
        raise ValueError(f"edge must be 'top' or 'bot', got {edge!r}")
    mid_x = float(xs[band].mean())
    mid_y = float(ys[band].mean())
    px, py = -vy, vx
    half = length / 2
    return (
        (mid_x - half * px, mid_y - half * py),
        (mid_x + half * px, mid_y + half * py),
    )


def cobb_between_axes(
    top_axis: tuple[float, float, float, float] | None,
    bottom_axis: tuple[float, float, float, float] | None,
) -> float | None:
    if top_axis is None or bottom_axis is None:
        return None
    _, _, tx, ty = top_axis
    _, _, bx, by = bottom_axis
    cosine = abs(tx * bx + ty * by)
    cosine = max(-1.0, min(1.0, cosine))
    return math.degrees(math.acos(cosine))


def spine_centroids(mask: np.ndarray) -> list[tuple[float, float]]:
    points = [centroid(mask[class_id]) for class_id in SPINE_IDS]
    filtered = [point for point in points if point is not None]
    filtered.sort(key=lambda point: point[1])
    return filtered


def fit_spine_polynomials(
    points: list[tuple[float, float]],
    degree: int = 4,
) -> tuple[np.ndarray, np.ndarray] | None:
    if len(points) < 3:
        return None
    array = np.asarray(points, dtype=np.float64)
    segment_lengths = np.linalg.norm(np.diff(array, axis=0), axis=1)
    if not np.all(segment_lengths > 0):
        keep = np.concatenate(([True], segment_lengths > 1e-6))
        array = array[keep]
        if len(array) < 3:
            return None
        segment_lengths = np.linalg.norm(np.diff(array, axis=0), axis=1)
    t = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    t = t / t[-1]
    clamped_degree = min(degree, len(array) - 1)
    return np.polyfit(t, array[:, 0], deg=clamped_degree), np.polyfit(t, array[:, 1], deg=clamped_degree)


def sample_polynomial_curve(px: np.ndarray, py: np.ndarray, samples: int = 200) -> list[tuple[float, float]]:
    ts = np.linspace(0.0, 1.0, samples)
    xs = np.polyval(px, ts)
    ys = np.polyval(py, ts)
    return [(float(x), float(y)) for x, y in zip(xs, ys)]


def curve_tangent(px: np.ndarray, py: np.ndarray, t: float) -> tuple[float, float] | None:
    dx = float(np.polyval(np.polyder(px), t))
    dy = float(np.polyval(np.polyder(py), t))
    norm = math.hypot(dx, dy)
    if norm == 0:
        return None
    return dx / norm, dy / norm


def project_to_curve(px: np.ndarray, py: np.ndarray, x: float, y: float, samples: int = 401) -> float:
    ts = np.linspace(0.0, 1.0, samples)
    xs = np.polyval(px, ts)
    ys = np.polyval(py, ts)
    index = int(np.argmin((xs - x) ** 2 + (ys - y) ** 2))
    lower = ts[max(0, index - 1)]
    upper = ts[min(samples - 1, index + 1)]
    refined_ts = np.linspace(lower, upper, 51)
    refined_xs = np.polyval(px, refined_ts)
    refined_ys = np.polyval(py, refined_ts)
    refined_index = int(np.argmin((refined_xs - x) ** 2 + (refined_ys - y) ** 2))
    return float(refined_ts[refined_index])


def severity_from_cobb(cobb: float | None) -> str:
    if cobb is None:
        return "unknown"
    if cobb < 50:
        return "negative"
    if cobb < 60:
        return "mild"
    if cobb < 70:
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
    endplate_length_frac: float = 0.35,
    endplate_color: tuple[int, int, int] = ENDPLATE_COLOR,
    axis_color: tuple[int, int, int] = AXIS_COLOR,
    point_color: tuple[int, int, int] = DOT_COLOR,
) -> tuple[Image.Image, float | None, str | None, str | None, int]:
    height, width = mask.shape[1:]
    image = rgb.convert("RGB").resize((width, height), Image.BILINEAR).copy()
    draw = ImageDraw.Draw(image)

    if points is None:
        points = spine_centroids(mask)

    spine_fit = fit_spine_polynomials(points) if len(points) >= 3 else None
    if spine_fit is not None:
        curve = sample_polynomial_curve(*spine_fit)
        draw.line([tuple(point) for point in curve], fill=point_color, width=line_width)
    elif len(points) >= 2:
        draw.line([tuple(point) for point in points], fill=point_color, width=line_width)

    for x, y in points:
        draw.ellipse(
            [(x - point_radius, y - point_radius), (x + point_radius, y + point_radius)],
            fill=point_color,
            outline=(0, 0, 0),
            width=1,
        )

    top = pick_anchor(mask, TOP_ANCHOR_PREFERENCE)
    bottom = pick_anchor(mask, BOT_ANCHOR_PREFERENCE)
    top_label = top[0] if top else None
    bottom_label = bottom[0] if bottom else None
    top_axis = top[1] if top else None
    bottom_axis = bottom[1] if bottom else None

    def curve_axis(centroid_xy: tuple[float, float]) -> tuple[float, float, float, float] | None:
        if spine_fit is None:
            return None
        px_curve, py_curve = spine_fit
        t = project_to_curve(px_curve, py_curve, centroid_xy[0], centroid_xy[1])
        tangent = curve_tangent(px_curve, py_curve, t)
        if tangent is None:
            return None
        return centroid_xy[0], centroid_xy[1], tangent[0], tangent[1]

    if top is not None:
        top_axis = curve_axis((top[1][0], top[1][1])) or top[1]
    if bottom is not None:
        bottom_axis = curve_axis((bottom[1][0], bottom[1][1])) or bottom[1]

    endplate_length = max(40.0, endplate_length_frac * width)

    if top is not None and top_axis is not None:
        segment = endplate_endpoints(mask[label2id[top[0]]], top_axis, edge="top", length=endplate_length)
        if segment is not None:
            draw.line(segment, fill=endplate_color, width=line_width)
            cx, cy, vx, vy = top_axis
            extent = endplate_length / 2
            draw.line([(cx - vx * extent, cy - vy * extent), (cx + vx * extent, cy + vy * extent)], fill=axis_color, width=max(1, line_width - 2))

    if bottom is not None and bottom_axis is not None:
        segment = endplate_endpoints(mask[label2id[bottom[0]]], bottom_axis, edge="bot", length=endplate_length)
        if segment is not None:
            draw.line(segment, fill=endplate_color, width=line_width)
            cx, cy, vx, vy = bottom_axis
            extent = endplate_length / 2
            draw.line([(cx - vx * extent, cy - vy * extent), (cx + vx * extent, cy + vy * extent)], fill=axis_color, width=max(1, line_width - 2))

    cobb = cobb_between_axes(top_axis, bottom_axis)
    if cobb is not None:
        size = font_size if font_size is not None else max(12, width // 22)
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", size)
        except Exception:
            font = ImageFont.load_default()
        anchors = ""
        if top_label and bottom_label:
            top_short = top_label.split("(")[-1].rstrip(")")
            bottom_short = bottom_label.split("(")[-1].rstrip(")")
            anchors = f"  {top_short.upper()}->{bottom_short.upper()}"
        text = f"Cobb={cobb:.2f}°  {severity_from_cobb(cobb).upper()}{anchors}"
        pad = 4
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.rectangle([(pad, pad), (pad + text_width + 2 * pad, pad + text_height + 2 * pad)], fill=(0, 0, 0))
        draw.text((pad * 2, pad * 2), text, fill=(255, 255, 255), font=font)

    return image, cobb, top_label, bottom_label, len(points)


def analyse_image(
    model: XAnatomy,
    image_path: Path,
    *,
    threshold: float,
    line_width: int,
    point_radius: int,
    endplate_color: tuple[int, int, int] = ENDPLATE_COLOR,
    axis_color: tuple[int, int, int] = AXIS_COLOR,
    point_color: tuple[int, int, int] = DOT_COLOR,
) -> dict:
    rgb = Image.open(image_path).convert("RGB")
    _, probs = model.predict(rgb, return_probs=True)
    mask = probs > threshold
    points = spine_centroids(mask)
    segmentation = render_segmentation(rgb, mask)
    interpretation, cobb, top_label, bottom_label, vertebrae_used = render_interpretation(
        rgb,
        mask,
        points=points,
        line_width=line_width,
        point_radius=point_radius,
        endplate_color=endplate_color,
        axis_color=axis_color,
        point_color=point_color,
    )
    return {
        "image_path": image_path,
        "segmentation": segmentation,
        "interpretation": interpretation,
        "cobb": cobb,
        "severity": severity_from_cobb(cobb),
        "top_anchor": top_label,
        "bot_anchor": bottom_label,
        "vertebrae_used": vertebrae_used,
    }


def write_diagnosis(
    out_path: Path,
    image_path: Path,
    cobb: float | None,
    severity: str,
    top_label: str | None,
    bottom_label: str | None,
    n_used: int,
) -> None:
    cobb_text = f"{cobb:.2f} deg" if cobb is not None else "unavailable"
    out_path.write_text(
        f"image: {image_path.name}\n"
        f"severity: {severity}\n"
        f"Cobb_angle: {cobb_text}\n"
        f"top_anchor: {top_label or 'none'}\n"
        f"bot_anchor: {bottom_label or 'none'}\n"
        f"vertebrae_detected: {n_used}\n",
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
    if result["cobb"] is not None:
        title += f"   Cobb={result['cobb']:.2f}°  ({result['severity']})"
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
    min_vertebrae: int = 5,
    keep_unspecified: bool = False,
    endplate_color: tuple[int, int, int] = ENDPLATE_COLOR,
    axis_color: tuple[int, int, int] = AXIS_COLOR,
    point_color: tuple[int, int, int] = DOT_COLOR,
) -> int:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    images = list_image_files(input_dir)
    if not images:
        print(f"[kyphosis] no images found under {input_dir}")
        return 0

    written = 0
    skipped = 0
    insufficient = 0
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
                endplate_color=endplate_color,
                axis_color=axis_color,
                point_color=point_color,
            )
        except Exception as exc:
            print(f"  [{index}/{len(images)}] {image_path.relative_to(input_dir)} - FAILED: {exc}")
            continue

        cobb_str = f"{result['cobb']:.2f}°" if result["cobb"] is not None else "n/a"
        pred_counts[result["severity"]] += 1
        gt = parse_gt_severity(image_path.parent / "status.txt", DISEASE)
        gt_label = gt if gt is not None else "no-gt"

        if not keep_unspecified and gt in (None, "unspecified"):
            unspecified_skipped += 1
            continue
        if min_vertebrae > 0 and result["vertebrae_used"] < min_vertebrae:
            insufficient += 1
            continue
        if filter_ and not is_acceptable(result["cobb"], gt, FILTER_BANDS, tolerate):
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
        write_diagnosis(
            case_dir / "diagnosis.txt",
            image_path,
            result["cobb"],
            result["severity"],
            result["top_anchor"],
            result["bot_anchor"],
            result["vertebrae_used"],
        )
        written += 1
        print(
            f"  [{index}/{len(images)}] {image_path.relative_to(input_dir)} -> "
            f"Cobb={cobb_str} ({result['severity']})"
        )

    print()
    print(f"[kyphosis] predicted severity counts (all {sum(pred_counts.values())} processed):")
    print(format_counts(PRED_ORDER, pred_counts))
    if filter_:
        print(f"[kyphosis] kept by GT severity ({sum(kept_by_gt.values())} cases):")
        print(format_counts(GT_ORDER, kept_by_gt))
        print(f"[kyphosis] skipped by GT severity ({sum(skipped_by_gt.values())} cases):")
        print(format_counts(GT_ORDER, skipped_by_gt))
    if insufficient:
        print(f"[kyphosis] skipped for insufficient vertebrae: {insufficient}")
    if unspecified_skipped:
        print(f"[kyphosis] skipped unspecified/no-gt: {unspecified_skipped}")
    return written
