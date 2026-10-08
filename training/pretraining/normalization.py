from collections.abc import Sequence
import torch
from beartype import beartype
from einops import rearrange, reduce
from jaxtyping import Float, jaxtyped
from torch import Tensor
from architecture.pretraining.normalization import AudioVisualNormalizer


@jaxtyped(typechecker=beartype)
def fit_audio_visual_normalizer(
    examples: Sequence[
        tuple[
            Float[Tensor, "batch time patch video_feature"],
            Float[Tensor, "batch time audio_feature"],
        ]
    ],
    minimum_standard_deviation: float,
) -> AudioVisualNormalizer:
    if not examples or minimum_standard_deviation <= 0.0:
        raise ValueError("Normalizer fitting requires examples and a positive minimum")
    video_samples = torch.cat(
        [
            rearrange(
                video_latents,
                "batch time patch feature -> (batch time patch) feature",
            )
            for video_latents, audio_latents in examples
        ]
    )
    audio_samples = torch.cat(
        [
            rearrange(
                audio_latents,
                "batch time feature -> (batch time) feature",
            )
            for video_latents, audio_latents in examples
        ]
    )
    video_mean = reduce(video_samples, "sample feature -> feature", "mean")
    audio_mean = reduce(audio_samples, "sample feature -> feature", "mean")
    video_variance = reduce(
        (video_samples - video_mean).square(),
        "sample feature -> feature",
        "mean",
    )
    audio_variance = reduce(
        (audio_samples - audio_mean).square(),
        "sample feature -> feature",
        "mean",
    )
    return AudioVisualNormalizer(
        video_mean,
        video_variance.sqrt().clamp_min(minimum_standard_deviation),
        audio_mean,
        audio_variance.sqrt().clamp_min(minimum_standard_deviation),
    )
