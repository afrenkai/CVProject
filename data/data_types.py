from pydantic import Field, NonNegativeFloat, NonNegativeInt, PositiveFloat, PositiveInt
from configs.config_types import FrozenModel
from data.control_types import ControlLayout, Opencs2InputMap


class SourceConfig(FrozenModel):
    root: str
    audio_root: str | None = None
    splits: str
    video: str | None = None
    samples: str | None = None
    audio: str | None = None
    actions: str | None = None
    weight: NonNegativeFloat = 0.0


class DatasetConfig(FrozenModel):
    sources: dict[str, SourceConfig] = Field(default_factory=dict)
    # mainly for debugging but I guess we could make a huge synth dataset
    synthetic: bool = False
    # better to do Path type here???
    latent_cache: str
    clips: PositiveInt
    validation_clips: NonNegativeInt = 0
    seed: NonNegativeInt = 0
    mirror_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    recorded_actions: bool = False
    workers: NonNegativeInt = 8
    # should this pass by cfg or just fail after like n=4 hardcoded?
    attempts: PositiveInt = 16


class ActionClipConfig(FrozenModel):
    context_frames: PositiveInt
    target_frames: PositiveInt
    video_frames_per_step: PositiveInt
    video_rate: PositiveInt
    control_state_lookback_seconds: PositiveFloat = 10.0
    maximum_clock_residual: PositiveFloat = 1 / 60
    maximum_screen_gap: PositiveFloat = 0.15
    control_layout: ControlLayout = Field(default_factory=ControlLayout)
    opencs2_inputs: Opencs2InputMap = Field(default_factory=Opencs2InputMap)
