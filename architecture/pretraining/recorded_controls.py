import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn
from train_datasets.control_types import ControlEncoderConfig


class RecordedControlEncoder(nn.Module):
    def __init__(self, config: ControlEncoderConfig, action_dimension: int) -> None:
        super().__init__()
        if len(config.axis_scales) != len(config.layout.axis_names):
            raise ValueError("Every recorded control axis requires a training-only scale")
        self.config = config
        self.button_count = config.layout.keyboard_count + len(config.layout.mouse_buttons)
        self.register_buffer("axis_scales", torch.tensor(config.axis_scales, dtype=torch.float32))
        self.network = nn.Sequential(
            nn.Linear(config.layout.dimension, config.hidden_dimension),
            nn.SiLU(),
            nn.Linear(config.hidden_dimension, action_dimension),
        )

    @jaxtyped(typechecker=beartype)
    def forward(self, controls: Float[Tensor, "batch time control"]) -> Float[Tensor, "batch time action"]:
        if controls.shape[-1] != self.config.layout.dimension:
            raise ValueError("Recorded controls do not match the checkpoint control layout")
        buttons = controls[..., : self.button_count]
        axes = torch.asinh(controls[..., self.button_count :] / self.axis_scales)
        return self.network(torch.cat((buttons, axes), dim=-1))
