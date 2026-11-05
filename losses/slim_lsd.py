import torch
from beartype import beartype
from einops import rearrange
from jaxtyping import Float, jaxtyped
from torch import Tensor
from torch.nn.attention import SDPBackend, sdpa_kernel
from models.world_model import AudioVisualFlowPredictor
from transport.flow import AudioVisualFlowTransport
from transport.solvers import flow_map_jump


# ALL FOR ABLATION, not in current loss_fn
@jaxtyped(typechecker=beartype)
def slim_lsd_loss(
    transport: AudioVisualFlowTransport,
    predictor: AudioVisualFlowPredictor,
    video_latents: Float[Tensor, "batch time patch video_feature"],
    audio_latents: Float[Tensor, "batch time audio_feature"],
    context_steps: int,
    actions: Float[Tensor, "batch target_time action"] | None,
    maximum_time_gap: float,
    context_actions: Float[Tensor, "batch context_time action"] | None = None,
    maximum_end_time: float = 0.95,
    adaptive_power: float = 0.75,
) -> tuple[Float[Tensor, ""], Float[Tensor, ""]]:
    context_video = video_latents[:, :context_steps].detach()
    context_audio = audio_latents[:, :context_steps].detach()
    target_video = transport.video_target(video_latents[:, context_steps:].detach(), context_video)
    target_audio = audio_latents[:, context_steps:].detach()
    interpolant = transport.sample_interpolant(target_video, target_audio)
    end_time = (
        transport.sample_end_time(interpolant.time, maximum_time_gap)
        .clamp(max=maximum_end_time)
        .maximum(interpolant.time)
    )
    detached_actions = None if actions is None else actions.detach()
    detached_context_actions = None if context_actions is None else context_actions.detach()

    student_video, student_audio = predictor(
        interpolant.video_state,
        interpolant.audio_state,
        interpolant.time,
        end_time,
        context_video,
        context_audio,
        detached_actions,
        detached_context_actions,
    )
    with torch.no_grad():
        with sdpa_kernel(SDPBackend.MATH):
            predictor_outputs, end_derivatives = torch.func.jvp(
                lambda differentiable_end: predictor(
                    interpolant.video_state,
                    interpolant.audio_state,
                    interpolant.time,
                    differentiable_end,
                    context_video,
                    context_audio,
                    detached_actions,
                    detached_context_actions,
                ),
                (end_time,),
                (torch.ones_like(end_time),),
            )
        predicted_video, predicted_audio = predictor_outputs
        video_end_derivative, audio_end_derivative = end_derivatives
        if predicted_video.shape != student_video.shape or predicted_audio.shape != student_audio.shape:
            raise RuntimeError("Slim-LSD JVP changed predictor output shapes")
        mapped_video, mapped_audio = flow_map_jump(
            interpolant.video_state,
            interpolant.audio_state,
            student_video.detach(),
            student_audio.detach(),
            interpolant.time,
            end_time,
        )
        teacher_video, teacher_audio = predictor(
            mapped_video,
            mapped_audio,
            end_time,
            end_time,
            context_video,
            context_audio,
            detached_actions,
            detached_context_actions,
        )

    tangent_video, tangent_audio = flow_map_jump(
        student_video,
        student_audio,
        video_end_derivative,
        audio_end_derivative,
        interpolant.time,
        end_time,
    )
    remaining = 1.0 - end_time
    video_residual = rearrange(remaining, "batch -> batch 1 1 1") * (tangent_video - teacher_video)
    audio_residual = rearrange(remaining, "batch -> batch 1 1") * (tangent_audio - teacher_audio)
    residual = video_residual.square().mean(dim=(1, 2, 3)) + audio_residual.square().mean(dim=(1, 2))
    adaptive_weight = (residual.detach() + 1e-3).pow(-adaptive_power)
    return torch.mean(adaptive_weight * residual), residual.detach().mean()
