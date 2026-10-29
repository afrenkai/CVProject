import math
import time
import torch
from beartype import beartype
from jaxtyping import Float, Int, jaxtyped
from torch import Tensor
from evaluation.evaluation_types import (
    ForecastReview,
    ForecastStatistics,
    ForecastSummary,
)
from training.pretraining.objective import PretrainingObjective
from transport.flow import AudioVisualFlowTransport
from transport.solvers import solve_flow_map
from transport.transport_types import ForecastConfig


class ForecastMoments:
    @jaxtyped(typechecker=beartype)
    def __init__(self, reference: Float[Tensor, "..."], persistence: Float[Tensor, "..."]) -> None:
        assert reference.shape == persistence.shape
        assert torch.isfinite(reference).all() and torch.isfinite(persistence).all()
        self.reference = reference.float()
        self.persistence = persistence.float()
        self.mean = torch.zeros_like(reference, dtype=torch.float32)
        self.squared_deviations = torch.zeros_like(reference, dtype=torch.float32)
        self.count = 0
        self.sample_error_sum = 0.0

    @jaxtyped(typechecker=beartype)
    def update(self, sample: Float[Tensor, "..."]) -> None:
        assert sample.shape == self.reference.shape and torch.isfinite(sample).all()
        value = sample.float()
        self.count += 1
        self.sample_error_sum += (value - self.reference).square().mean().item()
        delta = value - self.mean
        self.mean += delta / self.count
        self.squared_deviations += delta * (value - self.mean)

    def summary(self) -> ForecastStatistics:
        assert self.count > 0
        sample_error = self.sample_error_sum / self.count
        mean_error = (self.mean - self.reference).square().mean().item()
        variance = (self.squared_deviations / self.count).mean().item()
        assert math.isclose(sample_error, mean_error + variance, rel_tol=0.0001, abs_tol=0.00001)
        return ForecastStatistics(
            n_futures=self.count,
            mean_sample_mse=sample_error,
            sample_mean_mse=mean_error,
            population_sample_variance=variance,
            persistence_mse=(self.persistence - self.reference).square().mean().item(),
        )


@torch.no_grad()
@jaxtyped(typechecker=beartype)
def forecast_window(
    objective: PretrainingObjective,
    video: Float[Tensor, "one time patch video_feature"],
    audio: Float[Tensor, "one time audio_feature"],
    actions: Float[Tensor, "one time control"],
    action_present: Int[Tensor, "one"],  # noqa: F821
    example_seed: int,
    config: ForecastConfig,
    seed: int,
    review: ForecastReview | None = None,
) -> ForecastSummary:
    assert video.shape[0] == 1 and not objective.training
    context = objective.context_steps
    target_video, target_audio = video[:, context:], audio[:, context:]
    context_video, context_audio = video[:, :context], audio[:, :context]
    video_moments = ForecastMoments(target_video, context_video[:, -1:].expand_as(target_video))
    audio_moments = ForecastMoments(target_audio, context_audio[:, -1:].expand_as(target_audio))
    settings = objective.transport.config
    transport = AudioVisualFlowTransport(
        settings.model_copy(
            update={
                "time_distribution": settings.time_distribution.model_copy(update={"shift": config.grid_shift}),
            }
        )
    )
    with torch.autocast(video.device.type, dtype=torch.bfloat16, enabled=video.is_cuda):
        conditioned = objective.encode_actions(actions, action_present)
    seeds = []
    if video.is_cuda:
        torch.cuda.synchronize(video.device)
    started = time.perf_counter()
    for future in range(config.n_futures):
        draw_seed = seed + 100000 * example_seed + future
        seeds.append(draw_seed)
        generator = torch.Generator(device=video.device).manual_seed(draw_seed)
        video_noise = settings.noise_standard_deviation * torch.randn(
            target_video.shape, generator=generator, device=video.device
        )
        audio_noise = settings.noise_standard_deviation * torch.randn(
            target_audio.shape, generator=generator, device=video.device
        )
        with torch.autocast(video.device.type, dtype=torch.bfloat16, enabled=video.is_cuda):
            prediction_video, prediction_audio = solve_flow_map(
                objective.predictor,
                video_noise,
                audio_noise,
                context_video,
                context_audio,
                actions=conditioned[:, context:],
                context_actions=conditioned[:, :context],
                integration_grid=transport.time_grid(config.integration_steps, video_noise),
                diagonal=True,
            )
        prediction_video = transport.video_frames(prediction_video, context_video)
        video_moments.update(prediction_video)
        audio_moments.update(prediction_audio)
    if video.is_cuda:
        torch.cuda.synchronize(video.device)
    elapsed = time.perf_counter() - started
    if review is not None:
        review(video_moments.mean, audio_moments.mean)
    return ForecastSummary(
        n_futures=config.n_futures,
        integration_steps=config.integration_steps,
        grid_shift=config.grid_shift,
        video=video_moments.summary(),
        audio=audio_moments.summary(),
        seeds=seeds,
        recorded_action_conditioning=bool(action_present.item()),
        sampling_seconds=elapsed,
    )
