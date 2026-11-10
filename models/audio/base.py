from abc import ABC, abstractmethod
from typing import Self
import torch
from beartype import beartype
from einops import rearrange
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn
from models.encoder_types import FrozenEncoderConfig


class AudioEncoder(nn.Module, ABC):
    def __init__(self, config: FrozenEncoderConfig) -> None:
        super().__init__()
        self.config = config

    def freeze(self, device: torch.device) -> Self:
        return self.to(device=device, dtype=torch.float32).eval().requires_grad_(False)

    @abstractmethod
    def features(self, mono: Float[Tensor, "mono sample"]) -> Float[Tensor, "mono feature time"]: ...

    @torch.no_grad()
    @jaxtyped(typechecker=beartype)
    def forward(self, waveform: Float[Tensor, "batch 2 sample"]) -> Float[Tensor, "mono_batch feature time"]:
        assert torch.isfinite(waveform).all()
        features = self.features(rearrange(waveform.float(), "batch channel sample -> (batch channel) sample"))
        assert features.shape[1] == self.config.audio_feature_dimension and torch.isfinite(features).all()
        return features
