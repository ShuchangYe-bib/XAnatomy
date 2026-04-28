from __future__ import annotations

import argparse
from pathlib import Path

from xanatomy import XAnatomy


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch anatomy segmentation with XAnatomy.")
    parser.add_argument("input", type=Path, help="Input image or folder.")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Output directory.")
    parser.add_argument("--fmt", default="npz", choices=["npz", "npy"], help="Mask output format.")
    parser.add_argument("--device", default="auto", help="'cuda', 'cpu', or 'auto'.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Sigmoid threshold.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional checkpoint override.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    model = XAnatomy(device=args.device, threshold=args.threshold, checkpoint=args.checkpoint)

    if args.input.is_file():
        args.output.mkdir(parents=True, exist_ok=True)
        out_path = args.output / f"{args.input.stem}.{args.fmt}"
        mask = model.predict(args.input)
        from xanatomy.io import save_mask

        save_mask(mask, out_path, fmt=args.fmt)
        print(out_path)
        return 0

    if args.input.is_dir():
        written = model.predict_folder(args.input, args.output, fmt=args.fmt)
        print(f"wrote {len(written)} mask(s) to {args.output}")
        return 0

    raise SystemExit(f"input not found: {args.input}")


if __name__ == "__main__":
    raise SystemExit(main())
