from collections.abc import Sequence
from typing import cast

import timm
import torch
from beartype import beartype
from einops import rearrange, reduce
from jaxtyping import Float, jaxtyped
from timm.models.vision_transformer import VisionTransformer
from torch import Tensor, nn
from torch.nn import functional


class DinoVideoEncoder(nn.Module):
    def __init__(
        self,
        model_id: str,
        pretrained: bool,
        image_size: int,
        layer_numbers: Sequence[int],
        mean: Sequence[float],
        standard_deviation: Sequence[float],
    ) -> None:
        super().__init__()
        if not layer_numbers or min(layer_numbers) < 1:
            raise ValueError("DINO layer numbers are one-based and must be positive")
        if len(mean) != 3 or len(standard_deviation) != 3:
            raise ValueError("DINO normalization requires three channels")
        self.backbone = cast(
            VisionTransformer,
            timm.create_model(
                model_id,
                pretrained=pretrained,
                num_classes=0,
                img_size=image_size,
            ),
        )
        self.layer_indices = [layer_number - 1 for layer_number in layer_numbers]
        self.image_size = image_size
        self.register_buffer(
            "mean",
            torch.tensor(mean, dtype=torch.float32)[None, :, None, None],
            persistent=False,
        )
        self.register_buffer(
            "standard_deviation",
            torch.tensor(standard_deviation, dtype=torch.float32)[None, :, None, None],
            persistent=False,
        )
        self.requires_grad_(False)
        self.eval()

    @jaxtyped(typechecker=beartype)
    def forward(
        self, frames: Float[Tensor, "batch time channel height width"]
    ) -> Float[Tensor, "batch time patch feature"]:
        batch_size, time_steps = frames.shape[:2]
        images = rearrange(
            frames,
            "batch time channel height width -> (batch time) channel height width",
        )
        if images.shape[-2:] != (self.image_size, self.image_size):
            images = functional.interpolate(
                images,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
        images = (images - self.mean) / self.standard_deviation
        with torch.no_grad():
            layer_features = cast(
                list[Tensor],
                self.backbone.forward_intermediates(
                    images,
                    indices=self.layer_indices,
                    norm=True,
                    output_fmt="NLC",
                    intermediates_only=True,
                ),
            )
            patch_features = reduce(
                torch.stack(layer_features),
                "layer image patch feature -> image patch feature",
                "mean",
            )
        return rearrange(
            patch_features,
            "(batch time) patch feature -> batch time patch feature",
            batch=batch_size,
            time=time_steps,
        )
