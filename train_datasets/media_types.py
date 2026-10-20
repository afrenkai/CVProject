from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    NonNegativeFloat,
)
from torch import Tensor


class MediaRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    path: Path
    dataset: str
    sample_id: str
    start_seconds: NonNegativeFloat = 0.0
    split: Literal["train", "validation", "test"]
    controls: Path | None = None
    recording_id: str | None = None


@dataclass(frozen=True)
class AudioVisualClip:
    record: MediaRecord
    frames: Tensor
    waveform: Tensor
