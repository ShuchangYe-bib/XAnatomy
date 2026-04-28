"""
3D visualization helpers for per-label NIfTI masks.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from xanatomy.labels import colors, label2id, resolve


def _load_nifti(path: Path) -> tuple[np.ndarray, np.ndarray]:
    import nibabel as nib

    image = nib.load(str(path))
    return np.asarray(image.get_fdata()), image.affine


def _downsample(array: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return array
    return array[::factor, ::factor, ::factor]


def load_case(case_dir: str | Path) -> dict:
    """
    Load a case from `<case_dir>/volume.nii.gz` plus `<case_dir>/masks_3d/`.
    """
    case_dir = Path(case_dir)
    volume, affine = _load_nifti(case_dir / "volume.nii.gz")
    spacing = tuple(abs(float(value)) for value in np.diag(affine)[:3])
    return {
        "volume": volume.astype(np.float32),
        "affine": affine,
        "spacing": spacing,
        "mask_dir": case_dir / "masks_3d",
    }


def list_available_labels(case: dict) -> list[str]:
    return sorted(path.stem.replace(".nii", "") for path in case["mask_dir"].glob("*.nii.gz"))


def load_label_mask(case: dict, label: str) -> np.ndarray:
    path = case["mask_dir"] / f"{label}.nii.gz"
    if not path.is_file():
        raise FileNotFoundError(f"no mask for label {label!r} in {case['mask_dir']}")
    array, _ = _load_nifti(path)
    return array.astype(bool)


def _mesh_from_mask(mask: np.ndarray, spacing: tuple[float, float, float]):
    from skimage import measure

    verts, faces, _, _ = measure.marching_cubes(mask.astype(np.uint8), level=0.5, spacing=spacing)
    return verts, faces


def plot_3d(
    case: dict,
    labels: str | list[str] | list[int],
    *,
    downsample: int = 2,
    show_volume: bool = True,
    volume_clip: tuple[float, float] = (-1000.0, 400.0),
    opacity: float = 0.5,
    title: str | None = None,
    z_scale: float = 3.0,
):
    """
    Render an interactive Plotly figure for a set of labels or groups.
    """
    import plotly.graph_objects as go

    class_ids = resolve(labels)
    volume = case["volume"]
    volume_ds = _downsample(volume, downsample)
    spacing_x, spacing_y, spacing_z = case["spacing"]
    ds_spacing = (spacing_x * downsample, spacing_y * downsample, spacing_z * downsample)

    traces = []

    if show_volume:
        z_index = volume_ds.shape[2] // 2
        slab = np.clip(volume_ds[:, :, z_index], volume_clip[0], volume_clip[1])
        slab = (slab - volume_clip[0]) / (volume_clip[1] - volume_clip[0])
        x_coords, y_coords = np.meshgrid(
            np.arange(slab.shape[1]) * ds_spacing[1],
            np.arange(slab.shape[0]) * ds_spacing[0],
        )
        z_coords = np.full_like(x_coords, z_index * ds_spacing[2] * z_scale, dtype=float)
        traces.append(
            go.Surface(
                x=x_coords,
                y=y_coords,
                z=z_coords,
                surfacecolor=slab,
                colorscale="Greys",
                showscale=False,
                opacity=0.35,
                name="axial slice",
                hoverinfo="skip",
            )
        )

    id_to_label = {value: key for key, value in label2id.items()}
    for class_id in class_ids:
        label_name = id_to_label.get(class_id, f"class {class_id}")
        path = case["mask_dir"] / f"{label_name}.nii.gz"
        if not path.is_file():
            continue
        mask, _ = _load_nifti(path)
        mask_ds = _downsample(mask > 0, downsample)
        if not mask_ds.any():
            continue
        try:
            verts, faces = _mesh_from_mask(mask_ds, ds_spacing)
        except (RuntimeError, ValueError):
            continue
        red, green, blue = colors[class_id]
        traces.append(
            go.Mesh3d(
                x=verts[:, 0],
                y=verts[:, 1],
                z=verts[:, 2] * z_scale,
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                color=f"rgb({red},{green},{blue})",
                opacity=opacity,
                name=label_name,
                showlegend=True,
                hovertemplate=f"{label_name}<extra></extra>",
            )
        )

    figure = go.Figure(data=traces)
    figure.update_layout(
        title=title or f"{len(traces) - int(show_volume)} structures",
        scene=dict(
            xaxis=dict(title="x (mm)", showbackground=False),
            yaxis=dict(title="y (mm)", showbackground=False),
            zaxis=dict(title="z (mm)", showbackground=False),
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(itemsizing="constant"),
    )
    return figure
