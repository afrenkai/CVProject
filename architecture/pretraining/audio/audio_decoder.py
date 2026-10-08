from beartype import beartype
from einops import rearrange
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn


class CausalAudioDecoder(nn.Module):
    def __init__(
        self, latent_dimension: int, output_channels: int, samples_per_token: int
    ) -> None:
        super().__init__()
        if output_channels <= 0 or samples_per_token <= 0:
            raise ValueError("Audio decoder dimensions must be positive")
        self.output_channels = output_channels
        self.samples_per_token = samples_per_token
        self.output_projection = nn.Linear(
            latent_dimension, output_channels * samples_per_token
        )

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        latents: Float[Tensor, "batch time latent"],
    ) -> Float[Tensor, "batch channel sample"]:
        samples = self.output_projection(latents)
        return rearrange(
            samples,
            "batch time (channel sample) -> batch channel (time sample)",
            channel=self.output_channels,
            sample=self.samples_per_token,
        )
