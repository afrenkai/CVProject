from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, PositiveInt


class ControlLayout(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    keyboard_count: PositiveInt = 256
    mouse_buttons: tuple[str, ...] = ("left", "right", "middle", "x1", "x2")
    axis_names: tuple[str, ...] = ("raw_mouse_x", "raw_mouse_y", "wheel", "horizontal_wheel")

    @property
    def dimension(self) -> int:
        return self.keyboard_count + len(self.mouse_buttons) + len(self.axis_names)


class ControlEncoderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    layout: ControlLayout = Field(default_factory=ControlLayout)
    axis_scales: tuple[PositiveFloat, ...]
    hidden_dimension: PositiveInt = 384
    conditioning: Literal["recorded", "posterior_distilled"]
    distillation_weight: PositiveFloat = 0.25
