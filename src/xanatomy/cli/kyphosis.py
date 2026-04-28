from __future__ import annotations

import argparse
import sys
from pathlib import Path

from xanatomy import XAnatomy
from xanatomy.analysis.kyphosis import (
    AXIS_COLOR,
    DOT_COLOR,
    ENDPLATE_COLOR,
    analyse_image,
    parse_color,
    process_folder,
    show_single,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kyphosis screening with XAnatomy.")
    parser.add_argument("input", type=Path, help="Path to a lateral chest X-ray image or folder.")
    parser.add_argument("--output", "-o", type=Path, default=None, help="Output folder for folder mode.")
    parser.add_argument("--threshold", type=float, default=0.1, help="Sigmoid threshold for the segmentation mask.")
    parser.add_argument("--line-width", type=int, default=4, help="Line thickness in pixels.")
    parser.add_argument("--point-radius", type=int, default=5, help="Centroid point radius.")
    parser.add_argument("--device", default="auto", help="'cuda', 'cpu', or 'auto'.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional checkpoint override.")
    parser.add_argument("--filter", action="store_true", help="Folder mode only. Filter by status.txt severity.")
    parser.add_argument("--tolerate", type=float, default=0.0, help="Severity-band tolerance in degrees.")
    parser.add_argument("--min-vertebrae", type=int, default=5, help="Minimum detected vertebrae for folder mode.")
    parser.add_argument("--keep-unspecified", action="store_true", help="Keep cases with missing or unspecified GT severity.")
    parser.add_argument("--endplate-color", type=parse_color, default=ENDPLATE_COLOR, help="Color of endplate guide lines.")
    parser.add_argument("--axis-color", type=parse_color, default=AXIS_COLOR, help="Color of anchor long-axis guide lines.")
    parser.add_argument("--point-color", type=parse_color, default=DOT_COLOR, help="Color of centroid markers and spline.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.input.exists():
        print(f"[kyphosis] input not found: {args.input}", file=sys.stderr)
        return 2

    model = XAnatomy(device=args.device, threshold=args.threshold, checkpoint=args.checkpoint)
    if args.input.is_file():
        result = analyse_image(
            model,
            args.input,
            threshold=args.threshold,
            line_width=args.line_width,
            point_radius=args.point_radius,
            endplate_color=args.endplate_color,
            axis_color=args.axis_color,
            point_color=args.point_color,
        )
        cobb_str = f"{result['cobb']:.3f}" if result["cobb"] is not None else "n/a"
        print(f"Cobb={cobb_str}  severity={result['severity']}")
        show_single(result)
        return 0

    out_dir = args.output or args.input.with_name(args.input.name + "_results")
    written = process_folder(
        model,
        args.input,
        out_dir,
        threshold=args.threshold,
        line_width=args.line_width,
        point_radius=args.point_radius,
        filter_=args.filter,
        tolerate=args.tolerate,
        min_vertebrae=args.min_vertebrae,
        keep_unspecified=args.keep_unspecified,
        endplate_color=args.endplate_color,
        axis_color=args.axis_color,
        point_color=args.point_color,
    )
    print(f"[kyphosis] wrote results for {written} image(s) to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
