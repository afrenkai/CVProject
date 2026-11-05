import torch
from beartype import beartype
from einops import rearrange
from jaxtyping import Float, jaxtyped
from torch import Tensor
from models.guidance import GuidedPredictor
from models.world_model import AudioVisualFlowPredictor

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
    predictor: AudioVisualFlowPredictor | GuidedPredictor,
    video_state: Float[Tensor, "batch time patch video_feature"],
    audio_state: Float[Tensor, "batch time audio_feature"],
    context_video: Float[Tensor, "batch context_time patch video_feature"],
    context_audio: Float[Tensor, "batch context_time audio_feature"],
    actions: Float[Tensor, "batch time action"] | None,
    integration_grid: Float[Tensor, "grid"],  # noqa: F821
    diagonal: bool,
    context_actions: Float[Tensor, "batch context_time action"] | None = None,
    context_noise: Float[Tensor, "batch"] | None = None,  # noqa: F821
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
            context_noise=context_noise,
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
