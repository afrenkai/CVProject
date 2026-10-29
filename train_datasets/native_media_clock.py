import bisect
from fractions import Fraction
import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor
from train_datasets.corpus_types import MediaWindowConfig
from train_datasets.frame_index_types import FrameTimeline


def audio_grid_start(origin: float, requested: float, sample_rate: int) -> float:
    origin_time = Fraction(str(origin))
    requested_time = Fraction(str(requested))
    coordinate = (requested_time - origin_time) * sample_rate
    sample_index = -(-coordinate.numerator // coordinate.denominator)
    aligned = origin_time + Fraction(sample_index, sample_rate)
    assert 0 <= aligned - requested_time < Fraction(1, sample_rate)
    return float(aligned)


@jaxtyped(typechecker=beartype)
def assert_native_video_clock(
    pts: Float[Tensor, "frame"],  # noqa: F821
    durations: Float[Tensor, "frame"],  # noqa: F821
    queries: Float[Tensor, "frame"],  # noqa: F821
    config: MediaWindowConfig,
    timeline: FrameTimeline | None,
) -> None:
    assert torch.isfinite(pts).all() and torch.isfinite(durations).all()
    assert torch.all(pts[1:] >= pts[:-1]) and torch.all(pts <= queries + config.presentation_tolerance)
    if timeline is None:
        assert torch.all(queries < pts + durations + config.presentation_tolerance)
        return
    for index, query in enumerate(queries.tolist()):
        following = bisect.bisect_right(timeline.seconds, query)
        assert following > 0
        current = timeline.seconds[following - 1]
        stop = timeline.seconds[following] if following < len(timeline.seconds) else timeline.end_seconds
        assert abs(float(pts[index]) - current) <= config.presentation_tolerance
        assert current <= query + config.presentation_tolerance and query < stop + config.presentation_tolerance
        assert query - float(pts[index]) <= config.maximum_screen_gap + config.presentation_tolerance
