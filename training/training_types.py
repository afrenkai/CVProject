from typing import Literal
from pydantic import Field, NonNegativeFloat, NonNegativeInt, PositiveFloat, PositiveInt
from configs.config_types import FrozenModel
from data.data_types import DatasetConfig
from models.encoder_types import EncoderConfig
from models.model_types import WorldModelConfig

type OptimizerName = Literal["muon", "gns_muon", "gns_muon_ampere", "dion3", "adamw"]


class SlimLsdConfig(FrozenModel):
    start: NonNegativeInt = 0
    weight: NonNegativeFloat = 0.0
    maximum_gap: float = Field(default=1.0, gt=0.0, le=1.0)
    ramp_steps: PositiveInt = 200
    diagonal_fraction: float = Field(default=0.1, ge=0.0, le=1.0)
    maximum_end_time: float = Field(default=0.95, gt=0.0, le=1.0)
    adaptive_power: NonNegativeFloat = 0.75


class ObjectiveConfig(FrozenModel):
    mode: Literal["audio_visual", "video_only"] = "audio_visual"
    time_shift: PositiveFloat | None = None
    action_dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    context_noise: float = Field(default=0.0, ge=0.0, lt=1.0)
    slim_lsd: SlimLsdConfig = Field(default_factory=SlimLsdConfig)


class OptimizerConfig(FrozenModel):
    name: OptimizerName = "muon"
    learning_rate: PositiveFloat = 2e-4
    weight_decay: NonNegativeFloat = 0.1
    warmup_steps: NonNegativeInt = 2000
    final_learning_rate_fraction: float = Field(default=0.0, ge=0.0, le=1.0)
    gradient_clip: NonNegativeFloat = 1.0
    betas: tuple[PositiveFloat, PositiveFloat] = (0.9, 0.999)


class TrainerConfig(FrozenModel):
    output: str
    epochs: PositiveInt = 1
    batch_size: PositiveInt = 8
    gradient_accumulation: PositiveInt = 1
    precision: Literal["fp32", "bf16"] = "bf16"
    device: Literal["cuda", "cpu"] = "cuda"
    activation_checkpointing: bool = False
    ema_decay: float = Field(default=0.999, ge=0.0, lt=1.0)
    normalizer_clips: PositiveInt = 1024
    log_every: PositiveInt = 25
    eval_steps: PositiveInt = 500
    checkpoint_steps: PositiveInt = 250
    checkpoint_limit: PositiveInt = 2
    max_steps: NonNegativeInt = 0
    initialize_from: str | None = None
    wandb_project: str = "speakup-stage1"
    wandb_mode: Literal["online", "offline", "disabled"] = "offline"
    compile: bool = False
    recompile_limit: PositiveInt = 512


class PretrainConfig(FrozenModel):
    data: DatasetConfig
    encoder: EncoderConfig = Field(default_factory=EncoderConfig)
    model: WorldModelConfig
    objective: ObjectiveConfig = Field(default_factory=ObjectiveConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    trainer: TrainerConfig
