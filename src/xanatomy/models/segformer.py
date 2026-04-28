"""
SegFormer model builder.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

_VARIANTS = ("b0", "b1", "b2", "b3", "b4", "b5")


class SegFormerWrapper(nn.Module):
    """
    HuggingFace SegFormer wrapper that returns logits at input resolution.
    """

    def __init__(self, model: nn.Module, num_classes: int) -> None:
        super().__init__()
        self.model = model
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.model(pixel_values=x).logits
        if output.shape[2:] != x.shape[2:]:
            output = F.interpolate(output, size=x.shape[2:], mode="bilinear", align_corners=False)
        return output


def build_segformer(
    num_classes: int,
    variant: str = "b3",
    in_channels: int = 3,
) -> SegFormerWrapper:
    """
    Build a SegFormer matching the checkpoint architecture.
    """
    from transformers import SegformerForSemanticSegmentation

    normalized_variant = variant.lower().replace("mit-", "")
    if normalized_variant not in _VARIANTS:
        raise ValueError(f"variant must be one of {_VARIANTS}, got {variant!r}")

    model = SegformerForSemanticSegmentation.from_pretrained(
        f"nvidia/mit-{normalized_variant}",
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    )
    if in_channels != 3:
        model.segformer.encoder.patch_embed.projection = nn.Conv2d(
            in_channels,
            model.config.hidden_sizes[0],
            kernel_size=7,
            stride=4,
            padding=3,
        )
    return SegFormerWrapper(model, num_classes)
