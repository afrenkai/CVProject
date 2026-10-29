import math
from collections.abc import Iterable
import torch
from beartype import beartype
from einops import rearrange, reduce
from jaxtyping import Float, Int, jaxtyped
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted
from torch import Tensor
from architecture.pretraining.normalization import AudioVisualNormalizer


class FeatureStatistics:
    def __init__(self, minimum_standard_deviation: float) -> None:
        if not math.isfinite(minimum_standard_deviation) or minimum_standard_deviation <= 0:
            raise ValueError("Normalizer fitting requires a positive standard deviation floor")
        self.minimum_standard_deviation = minimum_standard_deviation
        self.video = StandardScaler()
        self.audio = StandardScaler()

    @jaxtyped(typechecker=beartype)
    def observe(
        self,
        video: Float[Tensor, "batch time patch video_feature"],
        audio: Float[Tensor, "batch time audio_feature"],
    ) -> None:
        assert torch.isfinite(video).all() and torch.isfinite(audio).all()
        self.video.partial_fit(
            rearrange(video.detach().cpu(), "batch time patch feature -> (batch time patch) feature").numpy()
        )
        self.audio.partial_fit(rearrange(audio.detach().cpu(), "batch time feature -> (batch time) feature").numpy())

    def normalizer(self) -> AudioVisualNormalizer:
        check_is_fitted(self.video)
        check_is_fitted(self.audio)
        normalizer = AudioVisualNormalizer(
            torch.tensor(self.video.mean_, dtype=torch.float32),
            torch.tensor(self.video.var_, dtype=torch.float32).sqrt().clamp_min(self.minimum_standard_deviation),
            torch.tensor(self.audio.mean_, dtype=torch.float32),
            torch.tensor(self.audio.var_, dtype=torch.float32).sqrt().clamp_min(self.minimum_standard_deviation),
        )
        assert all(torch.isfinite(value).all() for value in normalizer.buffers())
        return normalizer


class ActionStatistics:
    def __init__(self, axis_count: int) -> None:
        assert axis_count > 0
        self.squares = torch.zeros(axis_count, dtype=torch.float64)
        self.rows = 0

    @jaxtyped(typechecker=beartype)
    def observe(
        self,
        actions: Float[Tensor, "batch time action"],
        action_present: Int[Tensor, "batch"],  # noqa: F821
    ) -> None:
        assert torch.isfinite(actions).all() and actions.shape[-1] >= self.squares.numel()
        axes = actions[action_present.bool()][..., -self.squares.numel() :].detach().double().cpu()
        self.squares += reduce(axes.square(), "batch time motion -> motion", "sum")
        self.rows += axes.shape[0] * axes.shape[1]

    def scales(self) -> list[float]:
        if not self.rows:
            return torch.ones_like(self.squares).tolist()
        scales = (self.squares / self.rows).sqrt().clamp_min(1.0)
        assert torch.isfinite(scales).all()
        return scales.tolist()


@jaxtyped(typechecker=beartype)
def fit_audio_visual_normalizer(
    examples: Iterable[
        tuple[
            Float[Tensor, "batch time patch video_feature"],
            Float[Tensor, "batch time audio_feature"],
        ]
    ],
    minimum_standard_deviation: float,
) -> AudioVisualNormalizer:
    statistics = FeatureStatistics(minimum_standard_deviation)
    for video, audio in examples:
        statistics.observe(video, audio)
    return statistics.normalizer()
