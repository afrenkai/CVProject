from collections.abc import Callable
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable
from pydantic import PositiveInt
from torch import Tensor
from configs.types import FrozenModel

type EncodeFeatures = Callable[[Tensor, Tensor], tuple[Tensor, Tensor]]


@runtime_checkable
class PrefixEncoder(Protocol):
    def encode(self, frames: Tensor, waveform: Tensor) -> tuple[Tensor, Tensor]: ...

    def encode_observed(self, frames: Tensor, waveform: Tensor) -> tuple[Tensor, Tensor]: ...


class FrozenEncoderConfig(FrozenModel):
    video_model: str = "vit_large_patch16_dinov3.lvd1689m"
    video_cache_directory: Path | None = None
    video_layers: tuple[PositiveInt, ...] = (6, 12, 18, 24)
    video_feature_dimension: Literal[1024] = 1024
    image_size: PositiveInt = 128
    video_rate: Literal[30] = 30
    sample_rate: Literal[16000] = 16000
    video_frames_per_step: Literal[3] = 3
    context_frames: PositiveInt = 60
    audio_model_directory: Path
    audio_feature_dimension: Literal[1536] = 1536
    audio_native_frame_rate: Literal[25] = 25
    audio_frame_repeat: Literal[2] = 2
    audio_tokens_per_step: Literal[5] = 5
