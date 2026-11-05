import torch
from beartype import beartype
from einops import rearrange
from jaxtyping import Float, jaxtyped
from torch import Tensor
from losses.loss_types import FlowLossTerms
from models.world_model import AudioVisualFlowPredictor
from transport.flow import AudioVisualFlowTransport


@jaxtyped(typechecker=beartype)
def mean_square(residual: Float[Tensor, "batch *feature"]) -> Float[Tensor, ""]:  # noqa: F821
    return residual.float().square().mean()


@jaxtyped(typechecker=beartype)
def flow_matching_terms(
    transport: AudioVisualFlowTransport,
    predictor: AudioVisualFlowPredictor,
    video_latents: Float[Tensor, "batch time patch video_feature"],
    audio_latents: Float[Tensor, "batch time audio_feature"],
    context_steps: int,
    actions: Float[Tensor, "batch target_time action"] | None,
    video_loss_weight: float = 1.0,
    audio_loss_weight: float = 1.0,
    temporal_loss_weight: float = 0.0,
    context_actions: Float[Tensor, "batch context_time action"] | None = None,
    context_keep: Float[Tensor, "batch"] | None = None,  # noqa: F821
) -> FlowLossTerms:
    assert video_loss_weight > 0 and audio_loss_weight >= 0 and temporal_loss_weight >= 0
    assert 0 < context_steps < video_latents.shape[1]
    context_video = video_latents[:, :context_steps].detach()
    context_audio = audio_latents[:, :context_steps].detach()
    target_video = video_latents[:, context_steps:].detach()
    target_audio = audio_latents[:, context_steps:].detach()
    target_video = transport.video_target(target_video, context_video)
    interpolant = transport.sample_interpolant(target_video, target_audio)
    context_noise = None
    if transport.config.context_noise > 0:
        context_noise = transport.config.context_noise * torch.rand(
            context_video.shape[0], device=context_video.device, dtype=context_video.dtype
        )
        video_level = rearrange(context_noise, "batch -> batch 1 1 1")
        audio_level = rearrange(context_noise, "batch -> batch 1 1")
        context_video = (1.0 - video_level) * context_video + video_level * torch.randn_like(context_video)
        context_audio = (1.0 - audio_level) * context_audio + audio_level * torch.randn_like(context_audio)
    if context_keep is not None:
        context_video = context_video * rearrange(context_keep, "batch -> batch 1 1 1")
        context_audio = context_audio * rearrange(context_keep, "batch -> batch 1 1")
    video_velocity, audio_velocity = predictor(
        interpolant.video_state,
        interpolant.audio_state,
        interpolant.time,
        interpolant.time,
        context_video,
        context_audio,
        actions,
        context_actions=context_actions,
        context_noise=context_noise,
    )
    video_loss = mean_square(video_velocity.float() - interpolant.video_velocity.float())
    audio_loss = mean_square(audio_velocity.float() - interpolant.audio_velocity.float())
    video_temporal = video_loss.new_zeros(())
    audio_temporal = audio_loss.new_zeros(())
    if temporal_loss_weight > 0:
        assert target_video.shape[1] > 1
        remaining = 1.0 - interpolant.time.float()
        projected_video = (
            interpolant.video_state.float() + rearrange(remaining, "batch -> batch 1 1 1") * video_velocity.float()
        )
        projected_audio = (
            interpolant.audio_state.float() + rearrange(remaining, "batch -> batch 1 1") * audio_velocity.float()
        )
        video_temporal = mean_square(torch.diff(projected_video, dim=1) - torch.diff(target_video.float(), dim=1))
        audio_temporal = mean_square(torch.diff(projected_audio, dim=1) - torch.diff(target_audio.float(), dim=1))
    weighted_flow = video_loss_weight * video_loss + audio_loss_weight * audio_loss
    weighted_temporal = temporal_loss_weight * (video_temporal + audio_temporal)
    loss = weighted_flow + weighted_temporal
    assert loss.dtype == torch.float32 and torch.isfinite(loss)
    return FlowLossTerms(loss, video_loss, audio_loss, video_temporal, audio_temporal, weighted_flow, weighted_temporal)
