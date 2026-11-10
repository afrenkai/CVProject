from typing import Literal
from pydantic import Field, NonNegativeFloat, NonNegativeInt, PositiveFloat, PositiveInt
from configs.config_types import FrozenModel

CostModality = Literal["video", "audio", "audio_video"]


class ActionSearchConfig(FrozenModel):
    clips: PositiveInt = 64
    population: PositiveInt = 64
    candidate_batch: PositiveInt = 16
    elites: PositiveInt = 8
    iterations: PositiveInt = 4
    chunk_steps: PositiveInt = 4
    integration_steps: PositiveInt = 4
    momentum: float = Field(default=0.3, ge=0.0, lt=1.0)
    axis_deviation: PositiveFloat = 2.0
    minimum_press_rate: NonNegativeFloat = 0.002
    minimum_motion: NonNegativeFloat = 5.0
    seed: NonNegativeInt = 0
