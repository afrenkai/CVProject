from dataclasses import dataclass
from typing import Literal
from pydantic import (
    BaseModel,
    ConfigDict,
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
)
from torch import Tensor
from configs.config_types import FrozenModel


class ForecastConfig(FrozenModel):
    n_futures: PositiveInt = 8
    integration_steps: PositiveInt = 16
    grid_shift: PositiveFloat = 4.0


class TimeDistributionConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["uniform", "logit_normal"] = "logit_normal"
    shift: PositiveFloat = 4.0
    mean: float = 0.0
    standard_deviation: PositiveFloat = 1.0


class TransportConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    noise_standard_deviation: PositiveFloat = 1.0
    sde_diffusion_scale: NonNegativeFloat = 1.0
    time_distribution: TimeDistributionConfig = TimeDistributionConfig()
    video_target: Literal["frame", "change"] = "frame"


@dataclass(frozen=True)
class AudioVisualInterpolant:
    video_state: Tensor
    audio_state: Tensor
    video_velocity: Tensor
    audio_velocity: Tensor
    time: Tensor
