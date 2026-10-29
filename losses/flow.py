import torch
from beartype import beartype
from einops import rearrange, reduce
from jaxtyping import Float, jaxtyped
from torch import Tensor
from architecture.av_model import AudioVisualFlowPredictor
from losses.types import FlowLossTerms
from transport.flow import AudioVisualFlowTransport


@jaxtyped(typechecker=beartype)
def modality_mean_square(
    residual: Float[Tensor, "batch *feature"],  # noqa: F821
    presence: Float[Tensor, "batch 2"] | None,
    modality: int,
) -> Float[Tensor, ""]:
    per_sample = reduce(residual.float().square(), "batch ... -> batch", "mean")
    if presence is None:
        return per_sample.mean()
    weights = presence[:, modality].float()
    return (per_sample * weights).sum() / weights.sum().clamp_min(1)


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
    modality_presence: Float[Tensor, "batch 2"] | None = None,
    context_presence: Float[Tensor, "batch 2"] | None = None,
) -> FlowLossTerms:
    assert video_loss_weight > 0 and audio_loss_weight > 0 and temporal_loss_weight >= 0
    assert 0 < context_steps < video_latents.shape[1]
    context_video = video_latents[:, :context_steps].detach()
    context_audio = audio_latents[:, :context_steps].detach()
    target_video = video_latents[:, context_steps:].detach()
    target_audio = audio_latents[:, context_steps:].detach()
    interpolant = transport.sample_interpolant(target_video, target_audio)
    video_state = interpolant.video_state
    audio_state = interpolant.audio_state
    if modality_presence is not None:
        if not torch.all((modality_presence == 0) | (modality_presence == 1)) or not torch.all(
            modality_presence.sum(-1) > 0
        ):
            raise ValueError("Every example must contain at least one declared observed modality")
        video_mask = modality_presence[:, 0, None, None, None]
        audio_mask = modality_presence[:, 1, None, None]
        context_video = context_video * video_mask
        context_audio = context_audio * audio_mask
        video_state = video_state * video_mask
        audio_state = audio_state * audio_mask
    if context_presence is not None:
        context_video = context_video * context_presence[:, 0, None, None, None]
        context_audio = context_audio * context_presence[:, 1, None, None]
    video_velocity, audio_velocity = predictor(
        video_state,
        audio_state,
        interpolant.time,
        interpolant.time,
        context_video,
        context_audio,
        actions,
        context_actions=context_actions,
    )
    video_loss = modality_mean_square(video_velocity.float() - interpolant.video_velocity.float(), modality_presence, 0)
    audio_loss = modality_mean_square(audio_velocity.float() - interpolant.audio_velocity.float(), modality_presence, 1)
    video_temporal = video_loss.new_zeros(())
    audio_temporal = audio_loss.new_zeros(())
    if temporal_loss_weight > 0:
        assert transport.config.audio_delay == 0.0 and target_video.shape[1] > 1
        remaining = 1.0 - interpolant.time.float()
        projected_video = (
            interpolant.video_state.float() + rearrange(remaining, "batch -> batch 1 1 1") * video_velocity.float()
        )
        projected_audio = (
            interpolant.audio_state.float() + rearrange(remaining, "batch -> batch 1 1") * audio_velocity.float()
        )
        video_temporal = modality_mean_square(
            torch.diff(projected_video, dim=1) - torch.diff(target_video.float(), dim=1), modality_presence, 0
        )
        audio_temporal = modality_mean_square(
            torch.diff(projected_audio, dim=1) - torch.diff(target_audio.float(), dim=1), modality_presence, 1
        )
    weighted_flow = video_loss_weight * video_loss + audio_loss_weight * audio_loss
    weighted_temporal = temporal_loss_weight * (video_temporal + audio_temporal)
    loss = weighted_flow + weighted_temporal
    assert loss.dtype == torch.float32 and torch.isfinite(loss)
    return FlowLossTerms(loss, video_loss, audio_loss, video_temporal, audio_temporal, weighted_flow, weighted_temporal)
