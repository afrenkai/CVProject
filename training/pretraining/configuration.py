from pathlib import Path
from typing import Literal, Self
from pydantic import (
    Field,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    model_validator,
)
from architecture.pretraining.encoder_types import FrozenEncoderConfig
from architecture.pretraining.types import WorldModelConfig
from configs.types import FrozenModel
from losses.types import FlowLossConfig
from train_datasets.corpus_types import CorpusConfig
from transport.types import ForecastConfig, TransportConfig


class WindowFiles(FrozenModel):
    train: Path
    validation: Path


class NormalizationFiles(FrozenModel):
    latents: Path
    actions: Path


class PretrainingSchedule(ForecastConfig):
    checkpoint_steps: Literal[1000] = 1000
    validation_steps: Literal[2000] = 2000
    validation_blocks_training: Literal[False] = False
    validation_shard_windows: PositiveInt = 256
    validation_world_size: PositiveInt = 2


class TrainerSettings(FrozenModel):
    maximum_steps: PositiveInt
    batch_size: PositiveInt = 1
    accumulation_steps: PositiveInt = 1
    learning_rate: PositiveFloat = 0.0002
    workers: NonNegativeInt = 2
    logging_steps: PositiveInt = 10
    seed: NonNegativeInt = 67
    activation_checkpointing: bool = True
    raw_read_attempts: PositiveInt = 64


class AllocationRuntime(FrozenModel):
    stop_file: Path | None = None
    deadline_epoch: PositiveFloat | None = None
    graceful_stop_margin_seconds: int = Field(default=600, ge=600)


class PretrainingRuntime(AllocationRuntime):
    resume: Path | None = None
    pause_step: PositiveInt | None = None


class FrozenInputConfig(FrozenModel):
    data: CorpusConfig
    encoder: FrozenEncoderConfig

    @model_validator(mode="after")
    def align_encoder_inputs(self) -> Self:
        data = self.data
        encoder = self.encoder
        data_grid = (
            data.image_size,
            data.video_rate,
            data.sample_rate,
            data.video_frames_per_step,
            data.context_frames,
        )
        encoder_grid = (
            encoder.image_size,
            encoder.video_rate,
            encoder.sample_rate,
            encoder.video_frames_per_step,
            encoder.context_frames,
        )
        if data_grid != encoder_grid:
            raise ValueError("Media window and frozen encoder grids must agree")
        if data.context_frames % data.video_frames_per_step or data.target_frames % data.video_frames_per_step:
            raise ValueError("Observed and future frames must form complete audiovisual blocks")
        return self


class PretrainingConfig(FrozenInputConfig):
    output: Path
    windows: WindowFiles
    normalization: NormalizationFiles
    world_model: WorldModelConfig
    transport: TransportConfig
    loss: FlowLossConfig = Field(default_factory=FlowLossConfig)
    schedule: PretrainingSchedule = Field(default_factory=PretrainingSchedule)
    trainer: TrainerSettings
    runtime: PretrainingRuntime = Field(default_factory=PretrainingRuntime)

    @model_validator(mode="after")
    def align_audio_tokens(self) -> Self:
        tokens = self.world_model.audio_tokens
        if tokens.channels != 2 or tokens.frames_per_step != self.encoder.audio_tokens_per_step:
            raise ValueError("Predictor audio tokens must preserve the frozen stereo encoder grid")
        return self
