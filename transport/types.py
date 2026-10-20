from dataclasses import dataclass
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    PositiveFloat,
)
from torch import Tensor


class TimeDistributionConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["uniform", "logit_normal"] = "logit_normal"
    shift: PositiveFloat = 4.0
    mean: float = 0.0
    standard_deviation: PositiveFloat = 1.0


class TransportConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    noise_standard_deviation: PositiveFloat = 1.0
    audio_delay: float = Field(default=0.25, ge=0.0, lt=1.0)
    sde_diffusion_scale: NonNegativeFloat = 1.0
    time_distribution: TimeDistributionConfig = TimeDistributionConfig()


@dataclass(frozen=True)
class AudioVisualInterpolant:
    video_state: Tensor
    audio_state: Tensor
    video_velocity: Tensor
    audio_velocity: Tensor
    time: Tensor
