from typing import Literal, Self
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveFloat,
    PositiveInt,
    model_validator,
)

type ActionAudioConditioning = Literal["transition", "past_only"]
type FlowPrediction = Literal["velocity", "data"]


class HybridHistoryConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    block_pattern: tuple[Literal["kda", "softmax"], ...] = ("kda", "kda", "kda", "softmax")
    chunk_size: PositiveInt = 16
    minimum_log_decay: float = Field(default=-5.0, lt=0.0)


class TransformerStageConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    hidden_dimension: PositiveInt
    heads: PositiveInt
    layers: PositiveInt
    key_value_heads: PositiveInt | None = None

    @model_validator(mode="after")
    def validate_attention_heads(self) -> Self:
        if self.hidden_dimension % self.heads:
            raise ValueError("Transformer hidden dimension must be divisible by its head count")
        if self.key_value_heads is not None and self.heads % self.key_value_heads:
            raise ValueError("Transformer head count must be a multiple of its key-value head count")
        return self


class AudioTokenConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    channels: PositiveInt = 1
    frames_per_step: PositiveInt = 1

    @property
    def tokens_per_step(self) -> int:
        return self.channels * self.frames_per_step


class WorldModelConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    action_dimension: PositiveInt
    backbone_attention: Literal["quadratic", "linear", "hybrid_kda"] = "quadratic"
    noise_time_embedding: Literal["sinusoidal", "gaussian_fourier"] = "sinusoidal"
    prediction: FlowPrediction = "velocity"
    attention_window: PositiveInt | None = None
    spatial_grid: tuple[PositiveInt, PositiveInt] | None = None
    audio_tokens: AudioTokenConfig = Field(default_factory=AudioTokenConfig)
    hybrid_history: HybridHistoryConfig = Field(default_factory=HybridHistoryConfig)
    backbone: TransformerStageConfig
    diffusion_head: TransformerStageConfig
    mlp_ratio: PositiveFloat = 4.0
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
