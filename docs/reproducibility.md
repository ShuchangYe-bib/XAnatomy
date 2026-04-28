# Reproducibility Notes

## Recommended Local Layout

```text
XAnatomy_github/
├── artifacts/checkpoints/segformer_b3_best.pt
├── data/raw/
├── data/processed/
└── outputs/
```

## Checkpoints

The inference API defaults to:

```text
artifacts/checkpoints/segformer_b3_best.pt
```

Override with either:

- `XANATOMY_CHECKPOINT=/abs/path/to/checkpoint.pt`
- `XAnatomy(checkpoint="...")`

## Dataset Handling

Do not commit patient-derived raw data into this repository. Keep local copies
under `data/raw/` or use symlinks to your managed data store.

## Paper Figures

Recommended practice:

- generate figures into `outputs/figures/`
- export summary tables into `outputs/tables/`
- keep scripts deterministic and config-driven
- never treat notebook outputs as source of record
