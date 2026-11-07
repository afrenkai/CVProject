from dataclasses import dataclass
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveFloat,
)
from torch import Tensor


class TimeDistributionConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    shift: PositiveFloat = 4.0
    mean: float = 0.0
    standard_deviation: PositiveFloat = 1.0


class TransportConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    noise_standard_deviation: PositiveFloat = 1.0
    time_distribution: TimeDistributionConfig = TimeDistributionConfig()
    context_noise: float = Field(default=0.0, ge=0.0, lt=1.0)


@dataclass(frozen=True)
class AudioVisualInterpolant:
    video_state: Tensor
    audio_state: Tensor
    video_velocity: Tensor
    audio_velocity: Tensor
    time: Tensor
