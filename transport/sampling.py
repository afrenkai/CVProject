import torch
from beartype import beartype
from einops import repeat
from jaxtyping import Float, jaxtyped
from torch import Tensor
from architecture.av_model import AudioVisualFlowPredictor
from transport.flow import AudioVisualFlowTransport
from transport.solvers import solve_flow_map, solve_sde


@torch.no_grad()
@jaxtyped(typechecker=beartype)
def sample_chunk(
    transport: AudioVisualFlowTransport,
    predictor: AudioVisualFlowPredictor,
    context_video: Float[Tensor, "batch context_time patch video_feature"],
    context_audio: Float[Tensor, "batch context_time audio_feature"],
    actions: Float[Tensor, "batch target_time action"] | None,
    target_steps: int,
    integration_steps: int,
    diagonal: bool = False,
    stochastic: bool = False,
) -> tuple[
    Float[Tensor, "batch target_time patch video_feature"],
    Float[Tensor, "batch target_time audio_feature"],
]:
    if context_video.shape[:2] != context_audio.shape[:2]:
        raise ValueError("Transport audio and video context must align")
    if context_video.shape[1] == 0 or target_steps <= 0:
        raise ValueError("Sampling requires context and target steps")
    if actions is not None and actions.shape[:2] != (
        context_video.shape[0],
        target_steps,
    ):
        raise ValueError("Actions must align with the sampled chunk")

    video_state = transport.config.noise_standard_deviation * torch.randn_like(
        repeat(
            context_video[:, -1],
            "batch patch feature -> batch time patch feature",
            time=target_steps,
        )
    )
    audio_state = transport.config.noise_standard_deviation * torch.randn_like(
        repeat(
            context_audio[:, -1],
            "batch feature -> batch time feature",
            time=target_steps,
        )
    )
    integration_grid = transport.time_grid(integration_steps, context_video)
    if stochastic:
        return solve_sde(
            predictor,
            video_state,
            audio_state,
            context_video,
            context_audio,
            actions,
            integration_grid,
            transport.config.audio_delay,
            transport.config.noise_standard_deviation,
            transport.config.sde_diffusion_scale,
        )
    return solve_flow_map(
        predictor,
        video_state,
        audio_state,
        context_video,
        context_audio,
        actions,
        integration_grid,
        diagonal,
    )


@torch.no_grad()
@jaxtyped(typechecker=beartype)
def sample_rollout(
    transport: AudioVisualFlowTransport,
    predictor: AudioVisualFlowPredictor,
    context_video: Float[Tensor, "batch context_time patch video_feature"],
    context_audio: Float[Tensor, "batch context_time audio_feature"],
    actions: Float[Tensor, "batch rollout_time action"] | None,
    rollout_steps: int,
    chunk_steps: int,
    integration_steps: int,
    diagonal: bool = False,
    stochastic: bool = False,
) -> tuple[
    Float[Tensor, "batch rollout_time patch video_feature"],
    Float[Tensor, "batch rollout_time audio_feature"],
]:
    if context_video.shape[:2] != context_audio.shape[:2]:
        raise ValueError("Transport audio and video context must align")
    if context_video.shape[1] == 0 or rollout_steps <= 0 or chunk_steps <= 0:
        raise ValueError("Sampling requires context, rollout, and chunk steps")
    if actions is not None and actions.shape[:2] != (
        context_video.shape[0],
        rollout_steps,
    ):
        raise ValueError("Actions must align with the rollout")

    context_steps = context_video.shape[1]
    rolling_video = context_video
    rolling_audio = context_audio
    generated_video: list[Tensor] = []
    generated_audio: list[Tensor] = []
    for chunk_start in range(0, rollout_steps, chunk_steps):
        target_steps = min(chunk_steps, rollout_steps - chunk_start)
        chunk_actions = None if actions is None else actions[:, chunk_start : chunk_start + target_steps]
        video_chunk, audio_chunk = sample_chunk(
            transport,
            predictor,
            rolling_video,
            rolling_audio,
            chunk_actions,
            target_steps,
            integration_steps,
            diagonal,
            stochastic,
        )
        generated_video.append(video_chunk)
        generated_audio.append(audio_chunk)
        rolling_video = torch.cat((rolling_video, video_chunk), dim=1)[:, -context_steps:]
        rolling_audio = torch.cat((rolling_audio, audio_chunk), dim=1)[:, -context_steps:]
    return torch.cat(generated_video, dim=1), torch.cat(generated_audio, dim=1)
