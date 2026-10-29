import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor
from torch.nn.attention import SDPBackend, sdpa_kernel
from architecture.av_model import AudioVisualFlowPredictor
from architecture.pretraining.audio.audio_decoder import CausalAudioDecoder
from architecture.pretraining.video.video_decoder import DinoVideoDecoder
from losses.flow import flow_matching_terms
from transport.flow import AudioVisualFlowTransport
from transport.solvers import flow_map_jump


@jaxtyped(typechecker=beartype)
def audio_reconstruction_loss(
    decoder: CausalAudioDecoder,
    latents: Float[Tensor, "batch time latent"],
    waveform: Float[Tensor, "batch channel sample"],
) -> Float[Tensor, ""]:
    reconstruction = decoder(latents)
    if reconstruction.shape != waveform.shape:
        raise ValueError(f"Audio reconstruction shape {reconstruction.shape} does not match {waveform.shape}")
    return torch.mean((reconstruction - waveform).square())


@jaxtyped(typechecker=beartype)
def video_reconstruction_loss(
    decoder: DinoVideoDecoder,
    latents: Float[Tensor, "batch time patch feature"],
    frames: Float[Tensor, "batch time channel height width"],
) -> Float[Tensor, ""]:
    reconstruction = decoder(latents)
    if reconstruction.shape != frames.shape:
        raise ValueError("Video reconstruction and target shapes must match")
    return torch.mean((reconstruction - frames).square())


@jaxtyped(typechecker=beartype)
def flow_matching_loss(
    transport: AudioVisualFlowTransport,
    predictor: AudioVisualFlowPredictor,
    video_latents: Float[Tensor, "batch time patch video_feature"],
    audio_latents: Float[Tensor, "batch time audio_feature"],
    context_steps: int,
    actions: Float[Tensor, "batch target_time action"] | None,
    modality_presence: Float[Tensor, "batch 2"] | None = None,
    context_presence: Float[Tensor, "batch 2"] | None = None,
    context_actions: Float[Tensor, "batch context_time action"] | None = None,
) -> Float[Tensor, ""]:
    return flow_matching_terms(
        transport,
        predictor,
        video_latents,
        audio_latents,
        context_steps,
        actions,
        context_actions=context_actions,
        modality_presence=modality_presence,
        context_presence=context_presence,
    ).total


@jaxtyped(typechecker=beartype)
def slim_lsd_loss(
    transport: AudioVisualFlowTransport,
    predictor: AudioVisualFlowPredictor,
    video_latents: Float[Tensor, "batch time patch video_feature"],
    audio_latents: Float[Tensor, "batch time audio_feature"],
    context_steps: int,
    actions: Float[Tensor, "batch target_time action"] | None,
    maximum_time_gap: float,
) -> Float[Tensor, ""]:
    context_video = video_latents[:, :context_steps].detach()
    context_audio = audio_latents[:, :context_steps].detach()
    target_video = video_latents[:, context_steps:].detach()
    target_audio = audio_latents[:, context_steps:].detach()
    interpolant = transport.sample_interpolant(target_video, target_audio)
    end_time = transport.sample_end_time(interpolant.time, maximum_time_gap)
    detached_actions = None if actions is None else actions.detach()

    student_video, student_audio = predictor(
        interpolant.video_state,
        interpolant.audio_state,
        interpolant.time,
        end_time,
        context_video,
        context_audio,
        detached_actions,
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
        )

    tangent_video, tangent_audio = flow_map_jump(
        student_video,
        student_audio,
        video_end_derivative,
        audio_end_derivative,
        interpolant.time,
        end_time,
    )
    video_residual = tangent_video - teacher_video
    audio_residual = tangent_audio - teacher_audio
    return torch.mean(video_residual.square()) + torch.mean(audio_residual.square())
