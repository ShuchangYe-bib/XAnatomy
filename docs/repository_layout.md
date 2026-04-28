# Repository Layout

This rewrite intentionally separates four concerns that were mixed in the
original prototype:

1. `src/xanatomy/`
   Installable package code only.
2. `configs/`
   Stable, versioned configuration that should ship with the paper repo.
3. `data/` and `artifacts/`
   Local-only dataset and checkpoint locations.
4. `outputs/`
   Generated predictions, figures, and reports.

## Mapping From The Original Tree

- `XAnatomy/segmentor.py` -> `src/xanatomy/api.py`
- `XAnatomy/file_io.py` -> `src/xanatomy/io.py`
- `XAnatomy/label_mapper.py` -> `src/xanatomy/labels.py`
- `XAnatomy/visualize.py` -> `src/xanatomy/visualize.py`
- `XAnatomy/visualize3d.py` -> `src/xanatomy/visualize3d.py`
- `XAnatomy/models/segformer.py` -> `src/xanatomy/models/segformer.py`
- disease scripts -> `src/xanatomy/analysis/*.py` plus `src/xanatomy/cli/*.py`

## What Is Not Tracked

Large and mutable assets are intentionally kept out of version control:

- trained weights
- raw RadGenome-ChestCT data
- copied case folders
- generated overlays and diagnosis outputs
- ad hoc notebook outputs

That keeps the repo lightweight, reviewable, and reproducible.
