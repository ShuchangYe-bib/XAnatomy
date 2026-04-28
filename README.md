# XAnatomy

`XAnatomy` is a Python package for anatomy-aware chest X-ray segmentation and
anatomy-derived screening metrics.

This repository follows paper-reproducibility conventions:

- `src/` package layout for installable code
- `configs/` for model configuration
- `docs/` for reproducibility and project structure
- `tests/` for lightweight regression coverage
- externalized `data/`, `artifacts/`, and `outputs/` directories
- GitHub Actions CI for import and unit-test checks

## Repository Layout

```text
XAnatomy_github/
├── src/xanatomy/           # installable package
├── configs/                # model and runtime configuration
├── docs/                   # repo structure + reproducibility notes
├── tests/                  # lightweight unit tests
├── data/                   # dataset pointers only, no tracked raw data
├── artifacts/checkpoints/  # local model weights, gitignored
├── outputs/                # predictions and figures, gitignored
└── .github/workflows/      # CI
```

## Features

- multi-label chest X-ray anatomy segmentation with a SegFormer backbone
- visualization utilities for overlays, label groups, and legends
- CLI tools for batch segmentation and disease-specific screening
- 3D visualization support for case-level anatomy masks
- reproducible project layout with configs, tests, and CI

## Installation

```bash
cd XAnatomy_github
pip install -e .
```

Core dependencies are declared in [`pyproject.toml`](./pyproject.toml).

## Checkpoint Layout

The default runtime expects the trained weights at:

```text
artifacts/checkpoints/segformer_b3_best.pt
```

You can also override that path explicitly:

```python
from xanatomy import XAnatomy

model = XAnatomy(checkpoint="path/to/segformer_b3_best.pt")
```

Or via environment variable:

```bash
export XANATOMY_CHECKPOINT=/abs/path/to/segformer_b3_best.pt
```

## Quick Start

```python
from xanatomy import XAnatomy
from xanatomy.visualize import visualize_labels

model = XAnatomy(device="auto")
mask = model.predict("path/to/chest_pa.png")
overlay = visualize_labels(
    "path/to/chest_pa.png",
    mask,
    ["left lung", "right lung", "heart"],
    side_by_side=True,
)
overlay.save("outputs/example_overlay.png")
```

## CLI

```bash
xanatomy-segment path/to/image.png --output outputs/masks
xanatomy-cardiomegaly path/to/image.png
xanatomy-scoliosis path/to/image.png
xanatomy-kyphosis path/to/image.png
```

## Data Policy

This repository does not track:

- raw RadGenome-ChestCT data
- large example cases
- trained checkpoints
- generated evaluation outputs

See [data/README.md](./data/README.md),
[artifacts/checkpoints/README.md](./artifacts/checkpoints/README.md), and
[docs/reproducibility.md](./docs/reproducibility.md).

## Clinical Notes

Reference disease notes used by the screening heuristics are preserved under
[`docs/clinical_notes/`](./docs/clinical_notes/).

