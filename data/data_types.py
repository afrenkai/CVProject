import numpy as np
from pydantic import Field, NonNegativeFloat, NonNegativeInt, PositiveFloat, PositiveInt
from configs.config_types import FrozenModel
from data.control_types import ControlLayout, Opencs2InputMap


class SourceConfig(FrozenModel):
    root: str
    audio_root: str | None = None
    splits: str
    media: str | None = None
    samples: str | None = None
    audio: str | None = None
    ambisonic: str | None = None
    actions: str | None = None
    weight: NonNegativeFloat = 0.0


class DatasetConfig(FrozenModel):
    sources: dict[str, SourceConfig] = Field(default_factory=dict)
    # mainly for debugging but I guess we could make a huge synth dataset
    synthetic: bool = False
    # better to do Path type here???
    latent_cache: str | None = None
    cue_cache: str | None = None
    cue_period: int = Field(default=0, ge=0)
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


class CueConfig(FrozenModel):
    tick_rate: PositiveInt = 64
    context_seconds: PositiveFloat = 3.6
    target_seconds: PositiveFloat = 2.8
    history_seconds: NonNegativeFloat = 0.0
    onset_into_target: NonNegativeFloat = 1.0
    view_half_yaw: PositiveFloat = 53.0
    view_half_pitch: PositiveFloat = 30.0
    aim_degrees: PositiveFloat = 8.0
    engage_distance: PositiveFloat = 3000.0
    gunshot_distance: PositiveFloat = 3500.0
    footstep_distance: PositiveFloat = 1200.0
    running_speed: PositiveFloat = 150.0
    eye_height: PositiveFloat = 64.0
    quiet_gap_seconds: PositiveFloat = 4.0
    none_per_pov: PositiveInt = 2
    seed: int = 0


class RoundState(FrozenModel, arbitrary_types_allowed=True):
    ticks: np.ndarray
    position: np.ndarray
    yaw: np.ndarray
    pitch: np.ndarray
    team: np.ndarray
    alive: np.ndarray
    firing: np.ndarray
    running: np.ndarray
