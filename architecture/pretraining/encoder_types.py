from collections.abc import Callable
from pathlib import Path
from typing import Protocol, runtime_checkable
from pydantic import PositiveInt
from torch import Tensor
from configs.config_types import FrozenModel

type EncodeFeatures = Callable[[Tensor, Tensor], tuple[Tensor, Tensor]]


@runtime_checkable
class PrefixEncoder(Protocol):
    def encode(self, frames: Tensor, waveform: Tensor) -> tuple[Tensor, Tensor]: ...

    def encode_observed(
        self, frames: Tensor, waveform: Tensor
    ) -> tuple[Tensor, Tensor]: ...


class FrozenEncoderConfig(FrozenModel):
    video_model: str = "vit_large_patch16_dinov3.lvd1689m"
    video_cache_directory: Path | None = None
    video_layers: tuple[PositiveInt, ...] = (6, 12, 18, 24)
    video_feature_dimension: PositiveInt = 1024
    image_size: PositiveInt = 128
    video_rate: PositiveInt = 30
    sample_rate: PositiveInt = 16000
    video_frames_per_step: PositiveInt = 3
    context_frames: PositiveInt = 60
    audio_model_directory: Path
    audio_feature_dimension: PositiveInt = 1536
    audio_native_frame_rate: PositiveInt = 25
    audio_frame_repeat: PositiveInt = 2
    audio_tokens_per_step: PositiveInt = 5
    bfloat16: bool = False

    @property
    def video_latent_dimension(self) -> int:
        return self.video_feature_dimension * self.video_frames_per_step

    @property
    def audio_latent_dimension(self) -> int:
        return self.audio_feature_dimension * 2 * self.audio_tokens_per_step
