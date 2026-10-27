from dataclasses import dataclass
from typing import Protocol
from pydantic import NonNegativeFloat, PositiveFloat
from torch import Tensor
from configs.types import FrozenModel


class FlowLossConfig(FrozenModel):
    video_weight: PositiveFloat = 1.0
    audio_weight: PositiveFloat = 1.0
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

    def components(self) -> dict[str, Tensor]:
        return {
            "video_flow_loss": self.video_flow,
            "audio_flow_loss": self.audio_flow,
            "video_temporal_loss": self.video_temporal,
            "audio_temporal_loss": self.audio_temporal,
            "unweighted_flow_loss": self.video_flow + self.audio_flow,
            "weighted_flow_loss": self.weighted_flow,
            "weighted_temporal_loss": self.weighted_temporal,
        }


class LossReportingModel(Protocol):
    training_loss_sums: dict[str, Tensor]
    training_sample_count: int
