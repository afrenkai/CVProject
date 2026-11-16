from typing import Literal, Self
from pydantic import NonNegativeInt, PositiveFloat, PositiveInt, model_validator
from configs.config_types import FrozenModel

type AttentionBackend = Literal["sdpa", "flex", "flex_flash"]
type TokenLayout = tuple[int, int, int]


class TransformerStageConfig(FrozenModel):
    hidden_dimension: PositiveInt
    heads: PositiveInt
    layers: PositiveInt

    @model_validator(mode="after")
    def validate_attention_heads(self) -> Self:
        assert self.hidden_dimension % self.heads == 0
        return self


class AudioTokenConfig(FrozenModel):
    channels: PositiveInt = 1
    frames_per_step: PositiveInt = 1
    parts: PositiveInt = 1

    @property
    def tokens_per_step(self) -> int:
        return self.channels * self.frames_per_step * self.parts


class WorldModelConfig(FrozenModel):
    action_dimension: PositiveInt = 64
    attention_backend: AttentionBackend = "sdpa"
    spatial_grid: tuple[PositiveInt, PositiveInt] | None = None
    backbone: TransformerStageConfig
    diffusion_head: TransformerStageConfig
    mlp_ratio: PositiveFloat = 4.0
    mixture_of_transformers: bool = False
    fusion_tokens: NonNegativeInt = 0

    @model_validator(mode="after")
    def validate_mixture(self) -> Self:
        assert not self.mixture_of_transformers or self.spatial_grid is not None
        assert self.mixture_of_transformers or self.fusion_tokens == 0
        return self

    def token_layout(self, audio_tokens: AudioTokenConfig) -> TokenLayout | None:
        if not self.mixture_of_transformers or self.spatial_grid is None:
            return None
        return (self.spatial_grid[0] * self.spatial_grid[1], audio_tokens.tokens_per_step, self.fusion_tokens)
