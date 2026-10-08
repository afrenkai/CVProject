from collections.abc import Callable
from typing import Self
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveFloat,
    PositiveInt,
    model_validator,
)
from torch import Tensor


type RoPETuple = tuple[
    Callable[[Tensor], Tensor],
    Callable[[Tensor], Tensor],
]


class TransformerStageConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    hidden_dimension: PositiveInt
    heads: PositiveInt
    layers: PositiveInt

    @model_validator(mode="after")
    def validate_attention_heads(self) -> Self:
        if self.hidden_dimension % self.heads:
            raise ValueError(
                "Transformer hidden dimension must be divisible by its head count"
            )
        return self


class WorldModelConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    action_dimension: PositiveInt
    backbone: TransformerStageConfig
    diffusion_head: TransformerStageConfig
    mlp_ratio: PositiveFloat = 4.0
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
