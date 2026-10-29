import torch
from beartype import beartype
from einops import rearrange
from jaxtyping import Float, jaxtyped
from torch import Tensor
from architecture.av_model import AudioVisualFlowPredictor

# initial code taken from https://github.com/facebookresearch/Flow-World-Models/blob/main/transport/integrators.py


@jaxtyped(typechecker=beartype)
def flow_map_jump(
    video_state: Float[Tensor, "batch time patch video_feature"],
    audio_state: Float[Tensor, "batch time audio_feature"],
    video_velocity: Float[Tensor, "batch time patch video_feature"],
    audio_velocity: Float[Tensor, "batch time audio_feature"],
    start_time: Float[Tensor, "batch"],  # noqa: F821
    end_time: Float[Tensor, "batch"],  # noqa: F821
) -> tuple[
    Float[Tensor, "batch time patch video_feature"],
    Float[Tensor, "batch time audio_feature"],
]:
    if torch.any(end_time < start_time):
        raise ValueError("Flow-map end time must not precede start time")
    time_gap = end_time - start_time
    return (
        video_state + rearrange(time_gap, "batch -> batch 1 1 1") * video_velocity,
        audio_state + rearrange(time_gap, "batch -> batch 1 1") * audio_velocity,
    )


@torch.no_grad()
@jaxtyped(typechecker=beartype)
def solve_flow_map(
    predictor: AudioVisualFlowPredictor,
    video_state: Float[Tensor, "batch time patch video_feature"],
    audio_state: Float[Tensor, "batch time audio_feature"],
    context_video: Float[Tensor, "batch context_time patch video_feature"],
    context_audio: Float[Tensor, "batch context_time audio_feature"],
    actions: Float[Tensor, "batch time action"] | None,
    integration_grid: Float[Tensor, "grid"],  # noqa: F821
    diagonal: bool,
    context_actions: Float[Tensor, "batch context_time action"] | None = None,
) -> tuple[
    Float[Tensor, "batch time patch video_feature"],
    Float[Tensor, "batch time audio_feature"],
]:
    for integration_index in range(integration_grid.shape[0] - 1):
        start_time = integration_grid[integration_index].expand(video_state.shape[0])
        end_time = integration_grid[integration_index + 1].expand(video_state.shape[0])
        model_end_time = start_time if diagonal else end_time
        video_velocity, audio_velocity = predictor(
            video_state,
            audio_state,
            start_time,
            model_end_time,
            context_video,
            context_audio,
            actions,
            context_actions,
        )
        video_state, audio_state = flow_map_jump(
            video_state,
            audio_state,
            video_velocity,
            audio_velocity,
            start_time,
            end_time,
        )
    return video_state, audio_state


@torch.no_grad()
@jaxtyped(typechecker=beartype)
def solve_flow_map_given_audio(
    predictor: AudioVisualFlowPredictor,
    video_state: Float[Tensor, "batch time patch video_feature"],
    audio_noise: Float[Tensor, "batch time audio_feature"],
    target_audio: Float[Tensor, "batch time audio_feature"],
    context_video: Float[Tensor, "batch context_time patch video_feature"],
    context_audio: Float[Tensor, "batch context_time audio_feature"],
    actions: Float[Tensor, "batch time action"] | None,
    integration_grid: Float[Tensor, "grid"],  # noqa: F821
    context_actions: Float[Tensor, "batch context_time action"] | None = None,
) -> Float[Tensor, "batch time patch video_feature"]:
    for integration_index in range(integration_grid.shape[0] - 1):
        start_time = integration_grid[integration_index].expand(video_state.shape[0])
        end_time = integration_grid[integration_index + 1].expand(video_state.shape[0])
        audio_progress = rearrange(start_time, "batch -> batch 1 1")
        audio_state = (1.0 - audio_progress) * audio_noise + audio_progress * target_audio
        video_velocity, audio_velocity = predictor(
            video_state,
            audio_state,
            start_time,
            start_time,
            context_video,
            context_audio,
            actions,
            context_actions,
        )
        video_state, audio_state = flow_map_jump(
            video_state,
            audio_state,
            video_velocity,
            audio_velocity,
            start_time,
            end_time,
        )
    return video_state


@torch.no_grad()
@jaxtyped(typechecker=beartype)
def solve_sde(
    predictor: AudioVisualFlowPredictor,
    video_state: Float[Tensor, "batch time patch video_feature"],
    audio_state: Float[Tensor, "batch time audio_feature"],
    context_video: Float[Tensor, "batch context_time patch video_feature"],
    context_audio: Float[Tensor, "batch context_time audio_feature"],
    actions: Float[Tensor, "batch time action"] | None,
    integration_grid: Float[Tensor, "grid"],  # noqa: F821
    noise_standard_deviation: float,
    diffusion_scale: float,
) -> tuple[
    Float[Tensor, "batch time patch video_feature"],
    Float[Tensor, "batch time audio_feature"],
]:
    noise_variance = noise_standard_deviation**2
    epsilon = torch.finfo(video_state.dtype).eps
    for integration_index in range(integration_grid.shape[0] - 1):
        start_time = integration_grid[integration_index].expand(video_state.shape[0])
        end_time = integration_grid[integration_index + 1].expand(video_state.shape[0])
        video_velocity, audio_velocity = predictor(
            video_state,
            audio_state,
            start_time,
            start_time,
            context_video,
            context_audio,
            actions,
        )

        video_progress = rearrange(start_time, "batch -> batch 1 1 1")
        video_remaining = 1.0 - video_progress
        video_noise = video_state - video_progress * video_velocity
        video_score = -video_noise / (video_remaining.clamp_min(epsilon) * noise_variance)

        audio_progress = rearrange(start_time, "batch -> batch 1 1")
        audio_remaining = 1.0 - audio_progress
        audio_noise = audio_state - audio_progress * audio_velocity
        audio_score = -audio_noise / (audio_remaining.clamp_min(epsilon) * noise_variance)

        video_diffusion = diffusion_scale * video_remaining
        audio_diffusion = diffusion_scale * audio_remaining
        video_drift = video_velocity + 0.5 * video_diffusion.square() * video_score
        audio_drift = audio_velocity + 0.5 * audio_diffusion.square() * audio_score
        step_size = end_time - start_time
        video_step = rearrange(step_size, "batch -> batch 1 1 1")
        audio_step = rearrange(step_size, "batch -> batch 1 1")
        video_state = (
            video_state + video_step * video_drift + video_step.sqrt() * video_diffusion * torch.randn_like(video_state)
        )
        audio_state = (
            audio_state + audio_step * audio_drift + audio_step.sqrt() * audio_diffusion * torch.randn_like(audio_state)
        )
    return video_state, audio_state
