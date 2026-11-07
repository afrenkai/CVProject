import torch
from beartype import beartype
from einops import rearrange
from jaxtyping import Float, jaxtyped
from torch import Tensor
from models.world_model import AudioVisualFlowPredictor
from transport.flow import AudioVisualFlowTransport


@jaxtyped(typechecker=beartype)
def flow_matching_terms(
    transport: AudioVisualFlowTransport,
    predictor: AudioVisualFlowPredictor,
    video_latents: Float[Tensor, "batch time patch video_feature"],
    audio_latents: Float[Tensor, "batch time audio_feature"],
    context_steps: int,
    actions: Float[Tensor, "batch target_time action"],
    context_actions: Float[Tensor, "batch context_time action"],
    audio_weight: float,
) -> tuple[Tensor, Tensor, Tensor]:
    assert 0 < context_steps < video_latents.shape[1]
    context_video = video_latents[:, :context_steps].detach()
    context_audio = audio_latents[:, :context_steps].detach()
    target_video = transport.video_target(video_latents[:, context_steps:].detach(), context_video)
    interpolant = transport.sample_interpolant(target_video, audio_latents[:, context_steps:].detach())
    context_noise = None
    if transport.config.context_noise > 0:
        context_noise = transport.config.context_noise * torch.rand(
            context_video.shape[0], device=context_video.device, dtype=context_video.dtype
        )
        video_level = rearrange(context_noise, "batch -> batch 1 1 1")
        audio_level = rearrange(context_noise, "batch -> batch 1 1")
        context_video = (1.0 - video_level) * context_video + video_level * torch.randn_like(context_video)
        context_audio = (1.0 - audio_level) * context_audio + audio_level * torch.randn_like(context_audio)
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
    video_loss = (video_velocity.float() - interpolant.video_velocity.float()).square().mean()
    audio_loss = (audio_velocity.float() - interpolant.audio_velocity.float()).square().mean()
    loss = video_loss + audio_weight * audio_loss
    assert loss.dtype == torch.float32 and torch.isfinite(loss)
    return loss, video_loss, audio_loss
