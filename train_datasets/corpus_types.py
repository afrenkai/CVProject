from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from pydantic import (
    AliasChoices,
    Field,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
)
from torch import Tensor
from configs.config_types import FrozenModel
from train_datasets.control_types import ControlLayout
from train_datasets.input_types import InputStagingConfig


class SourceRecord(FrozenModel):
    source_id: str
    recording_id: str
    dataset: str
    path: Path
    actions_path: Path | None = Field(validation_alias=AliasChoices("actions_path", "controls_path"))
    audio_path: Path | None = None
    action_present: bool = True
    split: Literal["train", "validation", "test", "cs", "r6"]
    prior_split: str | None = None



class CorpusWindow(FrozenModel):
    source: SourceRecord
    start_seconds: float
    example_seed: NonNegativeInt = 0


class MediaWindowConfig(FrozenModel):
    frame_index_directory: Path | None = None
    input_staging: InputStagingConfig | None = None
    video_decoder_threads: PositiveInt = 2
    image_size: PositiveInt = 128
    video_rate: PositiveInt = 30
    sample_rate: PositiveInt = 16000
    context_frames: PositiveInt = 60
    target_frames: NonNegativeInt = 180
    presentation_tolerance: PositiveFloat = 0.002
    maximum_screen_gap: PositiveFloat = 0.15


class CorpusConfig(MediaWindowConfig):
    training_window_stride_seconds: PositiveFloat = 8.0
    control_state_lookback_seconds: PositiveFloat = 10.0
    video_frames_per_step: PositiveInt = 3
    maximum_clock_residual: PositiveFloat = 1 / 60
    control_layout: ControlLayout = Field(default_factory=ControlLayout)


@dataclass(frozen=True)
class DecodedAVWindow:
    frames: Tensor
    waveform: Tensor
    start_seconds: float
