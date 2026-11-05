from dataclasses import dataclass
from pydantic import NonNegativeFloat, PositiveFloat
from torch import Tensor
from configs.config_types import FrozenModel


class FlowLossConfig(FrozenModel):
    video_weight: PositiveFloat = 1.0
    audio_weight: NonNegativeFloat = 1.0
    temporal_weight: NonNegativeFloat = 0.0


@dataclass(frozen=True)
class FlowLossTerms:
    total: Tensor
    video_flow: Tensor
    audio_flow: Tensor
    video_temporal: Tensor
    audio_temporal: Tensor
    weighted_flow: Tensor
    weighted_temporal: Tensor
