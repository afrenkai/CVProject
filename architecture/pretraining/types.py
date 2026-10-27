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


class HybridHistoryConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    block_pattern: tuple[Literal["kda"], Literal["kda"], Literal["kda"], Literal["softmax"]] = (
        "kda",
        "kda",
        "kda",
        "softmax",
    )
    chunk_size: PositiveInt = 16
    minimum_log_decay: float = Field(default=-5.0, lt=0.0)


class TransformerStageConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    hidden_dimension: PositiveInt
    heads: PositiveInt
    layers: PositiveInt

    @model_validator(mode="after")
    def validate_attention_heads(self) -> Self:
        if self.hidden_dimension % self.heads:
            raise ValueError("Transformer hidden dimension must be divisible by its head count")
        return self


class WorldModelConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    action_dimension: PositiveInt
    backbone_attention: Literal["quadratic", "linear", "hybrid_kda"] = "quadratic"
    noise_time_embedding: Literal["sinusoidal", "gaussian_fourier"] = "sinusoidal"
    spatial_grid: tuple[PositiveInt, PositiveInt] | None = None
    hybrid_history: HybridHistoryConfig = Field(default_factory=HybridHistoryConfig)
    backbone: TransformerStageConfig
    diffusion_head: TransformerStageConfig
    mlp_ratio: PositiveFloat = 4.0
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
