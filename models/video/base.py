from abc import ABC, abstractmethod
from typing import Self
import torch
from beartype import beartype
from einops import rearrange
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn
from torch.nn import functional
from models.encoder_types import FrozenEncoderConfig


class VideoEncoder(nn.Module, ABC):
    def __init__(self, config: FrozenEncoderConfig) -> None:
        super().__init__()
        self.config = config

    def freeze(self, device: torch.device) -> Self:
        """
        detach from comp graph. lowk idk why we even need a docstring
        """
        return self.to(device=device, dtype=torch.float32).eval().requires_grad_(False)

    @abstractmethod
    def features(self, frames: Float[Tensor, "frame channel height width"]) -> Float[Tensor, "frame patch feature"]:
        """
        placeholder for the ABC, feats should arise from video encoder here naturally.
        """

    @torch.no_grad()
    @jaxtyped(typechecker=beartype)
    def forward(
        self, frames: Float[Tensor, "batch time channel height width"]
    ) -> Float[Tensor, "batch time patch feature"]:
        size = self.config.image_size
        flat = rearrange(frames.float(), "batch time channel height width -> (batch time) channel height width")
        if flat.shape[-2:] != (size, size):
            flat = functional.interpolate(flat, size=(size, size), mode="bilinear", align_corners=False)
        features = self.features(flat)
        assert features.shape[1:] == (self.config.video_patches, self.config.video.feature_dimension)
        return rearrange(features, "(batch time) patch feature -> batch time patch feature", batch=frames.shape[0])
