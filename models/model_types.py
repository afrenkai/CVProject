from typing import Literal, Self
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveFloat,
    PositiveInt,
    model_validator,
)

type FlowPrediction = Literal["velocity", "data"]
type AttentionBackend = Literal["sdpa", "flex", "flex_flash"]


class TransformerStageConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

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
    model_config = ConfigDict(frozen=True, extra="forbid")

    action_dimension: PositiveInt = 64
    noise_time_embedding: Literal["sinusoidal", "gaussian_fourier"] = "sinusoidal"
    prediction: FlowPrediction = "velocity"
    attention_window: PositiveInt | None = None
    attention_backend: AttentionBackend = "sdpa"
    context_noise_conditioning: bool = False
    spatial_grid: tuple[PositiveInt, PositiveInt] | None = None
    audio_tokens: AudioTokenConfig = Field(default_factory=AudioTokenConfig)
    backbone: TransformerStageConfig
    diffusion_head: TransformerStageConfig
    mlp_ratio: PositiveFloat = 4.0
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
