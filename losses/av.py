import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor
from torch.nn.attention import SDPBackend, sdpa_kernel
from architecture.av_model import AudioVisualFlowPredictor
from transport.flow import AudioVisualFlowTransport
from transport.solvers import flow_map_jump


@jaxtyped(typechecker=beartype)
def flow_matching_loss(
    transport: AudioVisualFlowTransport,
    predictor: AudioVisualFlowPredictor,
    video_latents: Float[Tensor, "batch time patch video_feature"],
    audio_latents: Float[Tensor, "batch time audio_feature"],
    context_steps: int,
    actions: Float[Tensor, "batch target_time action"] | None,
) -> Float[Tensor, ""]:
    """
    Follows the Lipman, et. al formulation of flow matching. However, the comp graph is slightly different

    effectively: model spits out audio and video latents. up to some context steps, context audio and video is retained. 

    these are stopgradded for the LSD formulation from Boffi, et. al

    the comp graph, as a result looks like

    Eq 27 from Boffi Et. Al for each modality. 


    @param transport: class which has sampling and solver for flow matching
    @param predictor: z_t predictor (DiTDH)which is our actual "world model" end state. this is supposed to do latent rollouts
    @param video_latents: float tensor representing the outputs of passing video through encoder. shape batch x timesteps x patch x video_feature (or feature, like we call it later)
    @param audio_latents float tensor representing results of passing waveform through audio encoder of shape batch x timesteps x audio_features
    @param context_steps: int. meant for a "window", or cutoff for how much ctx to provide. Will need to find happy medium for this and action conditioning. 
    @param actions: optional float tensor of shape batch x target_time x action meant to learn latent action for a LAM. might deprecate

    Returns:
        Float tensor with 1 entry, representing loss. video loss + audio loss
    """
    # ctx detach for rhs of flow matching
    context_video = video_latents[:, :context_steps].detach()
    context_audio = audio_latents[:, :context_steps].detach()
    
    #sg for LSD
    target_video = video_latents[:, context_steps:].detach()
    target_audio = audio_latents[:, context_steps:].detach()
    
    #(s,t)
    interpolant = transport.sample_interpolant(target_video, target_audio)
    #\dot(v) \forall{m}
    video_velocity, audio_velocity = predictor(
        interpolant.video_state,
        interpolant.audio_state,
        interpolant.time,
        interpolant.time,
        context_video,
        context_audio,
        actions,
    )

    # super easy MSE
    video_loss = torch.mean((video_velocity - interpolant.video_velocity).square())
    audio_loss = torch.mean((audio_velocity - interpolant.audio_velocity).square())
    return video_loss + audio_loss


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
    """
    See above loss, but uses the alternative sg formulation of https://arxiv.org/abs/2609.16222v1 for 1/2 the amount of NFEs during fwd prop. EMA teacher allows for self-distillation (obviously) but currently needs the MATH backend for SDPA which is NOT ideal.

    @param transport: class which has sampling and solver for flow matching
    @param predictor: z_t predictor (DiTDH)which is our actual "world model" end state. this is supposed to do latent rollouts
    @param video_latents: float tensor representing the outputs of passing video through encoder. shape batch x timesteps x patch x video_feature (or feature, like we call it later)
    @param audio_latents float tensor representing results of passing waveform through audio encoder of shape batch x timesteps x audio_features
    @param context_steps: int. meant for a "window", or cutoff for how much ctx to provide. Will need to find happy medium for this and action conditioning. 
    @param maximum_time_gap: how much the space between (u,t) can be in the SI since there is a chance the distribution becomes non-periodic or looses concavity
    """
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
    # TODO: replace with custom kernel later when have time, i dont think FA2 has jvp
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


@jaxtyped(typechecker=beartype)
def latent_prediction_errors(
    predicted_video: Float[Tensor, "batch time patch video_feature"],
    predicted_audio: Float[Tensor, "batch time audio_feature"],
    target_video: Float[Tensor, "batch time patch video_feature"],
    target_audio: Float[Tensor, "batch time audio_feature"],
) -> tuple[Float[Tensor, ""], Float[Tensor, ""], Float[Tensor, ""]]:
    video_error = torch.mean((predicted_video - target_video).square())
    audio_error = torch.mean((predicted_audio - target_audio).square())
    return video_error + audio_error, video_error, audio_error
