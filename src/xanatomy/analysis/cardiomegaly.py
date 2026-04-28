"""
Cardiomegaly analysis derived from anatomy segmentation.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from xanatomy import XAnatomy
from xanatomy.analysis.common import (
    copy_case_siblings,
    format_counts,
    is_acceptable,
    largest_component,
    list_image_files,
    parse_gt_severity,
)
from xanatomy.labels import label2id
from xanatomy.visualize import visualize_labels

DISEASE = "cardiomegaly"
FILTER_BANDS: dict[str, tuple[float, float]] = {
    "negative": (float("-inf"), 0.50),
    "mild": (0.50, 0.55),
    "moderate": (0.55, 0.60),
    "severe": (0.60, float("inf")),
}
PRED_ORDER = ["negative", "mild", "moderate", "severe", "extreme", "unknown"]
GT_ORDER = ["negative", "mild", "moderate", "severe", "unspecified", "no-gt"]
GENERATED_NAMES = {"segmentation.png", "interpretation.png", "diagnosis.txt"}
CLEAN_LABELS = ["heart", "left lung", "right lung"]
HEART_ID = label2id["heart"]
LLUNG_ID = label2id["left lung"]
RLUNG_ID = label2id["right lung"]
ORANGE = (37, 244, 238)
TEAL = (254, 44, 85)


def clean_mask(mask: np.ndarray, labels: list[str]) -> np.ndarray:
    output = mask.copy()
    for label in labels:
        class_id = label2id[label]
        output[class_id] = largest_component(output[class_id])
    return output


def widest_row(mask_2d: np.ndarray) -> tuple[int, int, int] | None:
    if not mask_2d.any():
        return None
    best: tuple[int, int, int, int] | None = None
    for y in np.where(np.any(mask_2d, axis=1))[0]:
        xs = np.where(mask_2d[y])[0]
        width = int(xs[-1] - xs[0])
        if best is None or width > best[0]:
            best = (width, int(y), int(xs[0]), int(xs[-1]))
    if best is None:
        return None
    return best[1], best[2], best[3]


def severity_from_ctr(ctr: float | None) -> str:
    if ctr is None:
        return "unknown"
    if ctr <= 0.50:
        return "negative"
    if ctr <= 0.55:
        return "mild"
    if ctr <= 0.60:
        return "moderate"
    if ctr <= 0.80:
        return "severe"
    return "extreme"


def render_segmentation(rgb: Image.Image, mask: np.ndarray, alpha: float = 0.35) -> Image.Image:
    return visualize_labels(rgb, mask, CLEAN_LABELS, alpha=alpha)


def render_interpretation(
    rgb: Image.Image,
    mask: np.ndarray,
    *,
    line_width: int = 4,
    point_radius: int = 7,
    font_size: int | None = None,
) -> tuple[Image.Image, float | None]:
    height, width = mask.shape[1:]
    image = rgb.convert("RGB").resize((width, height), Image.BILINEAR).copy()
    draw = ImageDraw.Draw(image)

    heart = widest_row(mask[HEART_ID])
    thorax = widest_row(mask[LLUNG_ID] | mask[RLUNG_ID])

    def draw_segment(coords: tuple[int, int, int], color: tuple[int, int, int]) -> None:
        y, x_left, x_right = coords
        draw.line([(x_left, y), (x_right, y)], fill=color, width=line_width)
        for x in (x_left, x_right):
            draw.ellipse(
                [
                    (x - point_radius, y - point_radius),
                    (x + point_radius, y + point_radius),
                ],
                fill=color,
                outline=(255, 255, 255),
                width=1,
            )

    if thorax is not None:
        draw_segment(thorax, TEAL)
    if heart is not None:
        draw_segment(heart, ORANGE)

    ctr = None
    if heart is not None and thorax is not None and (thorax[2] - thorax[1]) > 0:
        ctr = (heart[2] - heart[1]) / (thorax[2] - thorax[1])

    if ctr is not None:
        size = font_size if font_size is not None else max(12, width // 22)
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", size)
        except Exception:
            font = ImageFont.load_default()
        text = f"CTR={ctr:.4f}  {severity_from_ctr(ctr).upper()}"
        pad = 4
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.rectangle([(pad, pad), (pad + text_width + 2 * pad, pad + text_height + 2 * pad)], fill=(0, 0, 0))
        draw.text((pad * 2, pad * 2), text, fill=(255, 255, 255), font=font)

    return image, ctr


def analyse_image(
    model: XAnatomy,
    image_path: Path,
    *,
    threshold: float,
    line_width: int,
    point_radius: int,
) -> dict:
    rgb = Image.open(image_path).convert("RGB")
    _, probs = model.predict(rgb, return_probs=True)
    mask = clean_mask(probs > threshold, CLEAN_LABELS)
    segmentation = render_segmentation(rgb, mask)
    interpretation, ctr = render_interpretation(
        rgb,
        mask,
        line_width=line_width,
        point_radius=point_radius,
    )
    severity = severity_from_ctr(ctr)
    return {
        "image_path": image_path,
        "segmentation": segmentation,
        "interpretation": interpretation,
        "ctr": ctr,
        "severity": severity,
    }


def write_diagnosis(out_path: Path, image_path: Path, ctr: float | None, severity: str) -> None:
    ctr_text = f"{ctr:.4f}" if ctr is not None else "unavailable"
    out_path.write_text(
        f"image: {image_path.name}\n"
        f"severity: {severity}\n"
        f"CTR: {ctr_text}\n"
        "reference: Normal <=0.50, Mild 0.50-0.55, Moderate 0.55-0.60, "
        "Severe 0.60-0.80, Extreme >0.80\n",
        encoding="utf-8",
    )


def show_single(result: dict) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 6))
    axes[0].imshow(result["segmentation"])
    axes[0].set_title("Segmentation (heart + lungs)", fontsize=12, fontweight="bold")
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    title = "Interpretation"
    if result["ctr"] is not None:
        title += f"   CTR={result['ctr']:.3f}  ({result['severity']})"
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
    keep_unspecified: bool = False,
) -> int:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    images = list_image_files(input_dir)
    if not images:
        print(f"[cardiomegaly] no images found under {input_dir}")
        return 0

    print(f"[cardiomegaly] found {len(images)} images under {input_dir}")
    written = 0
    skipped = 0
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
            )
        except Exception as exc:
            print(f"  [{index}/{len(images)}] {image_path.relative_to(input_dir)} - FAILED: {exc}")
            continue

        ctr_str = f"{result['ctr']:.3f}" if result["ctr"] is not None else "n/a"
        pred_counts[result["severity"]] += 1

        gt = parse_gt_severity(image_path.parent / "status.txt", DISEASE)
        gt_label = gt if gt is not None else "no-gt"

        if not keep_unspecified and gt in (None, "unspecified"):
            unspecified_skipped += 1
            print(f"  [{index}/{len(images)}] SKIP {image_path.relative_to(input_dir)} (gt={gt_label})")
            continue

        if filter_ and not is_acceptable(result["ctr"], gt, FILTER_BANDS, tolerate):
            skipped += 1
            skipped_by_gt[gt_label] += 1
            print(
                f"  [{index}/{len(images)}] SKIP {image_path.relative_to(input_dir)} "
                f"(gt={gt_label}, CTR={ctr_str}, predicted={result['severity']})"
            )
            continue

        if filter_:
            kept_by_gt[gt_label] += 1

        case_dir = output_dir / relative_parent
        case_dir.mkdir(parents=True, exist_ok=True)
        copy_case_siblings(image_path.parent, case_dir, GENERATED_NAMES)
        result["segmentation"].save(case_dir / "segmentation.png")
        result["interpretation"].save(case_dir / "interpretation.png")
        write_diagnosis(case_dir / "diagnosis.txt", image_path, result["ctr"], result["severity"])
        written += 1
        print(f"  [{index}/{len(images)}] {image_path.relative_to(input_dir)} -> CTR={ctr_str} ({result['severity']})")

    print()
    print(f"[cardiomegaly] predicted severity counts (all {sum(pred_counts.values())} processed):")
    print(format_counts(PRED_ORDER, pred_counts))
    if filter_:
        print(f"[cardiomegaly] kept by GT severity ({sum(kept_by_gt.values())} cases):")
        print(format_counts(GT_ORDER, kept_by_gt))
        print(f"[cardiomegaly] skipped by GT severity ({sum(skipped_by_gt.values())} cases):")
        print(format_counts(GT_ORDER, skipped_by_gt))
    if unspecified_skipped:
        print(f"[cardiomegaly] skipped unspecified/no-gt: {unspecified_skipped}")
    return written
