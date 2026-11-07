import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn


class FrozenChannelNormalizer(nn.Module):
    @jaxtyped(typechecker=beartype)
    def __init__(
        self,
        mean: Float[Tensor, "channel"],  # noqa: F821
        standard_deviation: Float[Tensor, "channel"],  # noqa: F821
    ) -> None:
        super().__init__()
        if mean.shape != standard_deviation.shape:
            raise ValueError("Mean and standard deviation shapes differ")
        # I thikn this is still the way to check if anything in tensor adheres to some condition, maybe torch.where is better?
        if torch.any(standard_deviation <= 0.0):
            raise ValueError("Every channel must have positive standard deviation")
        self.register_buffer("mean", mean.detach().clone())
        self.register_buffer("standard_deviation", standard_deviation.detach().clone())

    @jaxtyped(typechecker=beartype)
    def forward(self, latents: Float[Tensor, "*tokens channel"]) -> Float[Tensor, "*tokens channel"]:
        # eff z -scaling (x - mu) / sqrt(sigma**2)
        return (latents - self.mean) / self.standard_deviation


class AudioVisualNormalizer(nn.Module):
    def __init__(
        self,
        video_mean: Tensor,
        video_standard_deviation: Tensor,
        audio_mean: Tensor,
        audio_standard_deviation: Tensor,
    ) -> None:
        """
        # wrapper around the nonsense I wrote on top.
        basically takes in mu_v, std_v, mu_a, std_a
        and uses the methods from the FrozenChannelNormalizer to give audio and video
        capability to be normed
        """
        super().__init__()
        self.video = FrozenChannelNormalizer(video_mean, video_standard_deviation)
        self.audio = FrozenChannelNormalizer(audio_mean, audio_standard_deviation)

    @jaxtyped(typechecker=beartype)
    def normalize_video(
        self, latents: Float[Tensor, "*tokens video_feature"]
    ) -> Float[Tensor, "*tokens video_feature"]:
        return self.video(latents)

    @jaxtyped(typechecker=beartype)
    def normalize_audio(
        self, latents: Float[Tensor, "*tokens audio_feature"]
    ) -> Float[Tensor, "*tokens audio_feature"]:
        return self.audio(latents)


class RunningMoments:
    def __init__(self) -> None:
        self.total: Tensor | None = None
        self.squared: Tensor | None = None
        self.count = 0

    def add(self, latents: Tensor, reduced_dims: tuple[int, ...]) -> None:
        values = latents.double()
        if self.total is None:
            self.total = torch.zeros(values.shape[-1], dtype=torch.float64, device=values.device)
            self.squared = torch.zeros_like(self.total)
        self.total += values.sum(dim=reduced_dims)
        self.squared += values.square().sum(dim=reduced_dims)
        self.count += values.numel() // values.shape[-1]

    def result(self) -> tuple[Tensor, Tensor]:
        mean = self.total / self.count
        return mean.float(), (self.squared / self.count - mean.square()).clamp_min(1e-8).sqrt().float()
