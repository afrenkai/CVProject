import math
import torch
from beartype import beartype
from einops import rearrange
from jaxtyping import Float, jaxtyped
from torch import Tensor
from architecture.av_model import AudioVisualFlowPredictor


def rademacher(shape: torch.Size, generator: torch.Generator, device: torch.device) -> Tensor:
    return (torch.randint(0, 2, shape, generator=generator) * 2 - 1).float().to(device)


def gaussian_negative_log_likelihood(state: Tensor, standard_deviation: float) -> Tensor:
    variance = standard_deviation**2
    return 0.5 * (state.square() / variance + math.log(2 * math.pi * variance)).flatten(1).sum(dim=1)


@jaxtyped(typechecker=beartype)
def flow_negative_log_likelihood(
    predictor: AudioVisualFlowPredictor,
    target_video: Float[Tensor, "batch time patch video_feature"],
    target_audio: Float[Tensor, "batch time audio_feature"],
    context_video: Float[Tensor, "batch context_time patch video_feature"],
    context_audio: Float[Tensor, "batch context_time audio_feature"],
    actions: Float[Tensor, "batch time action"] | None,
    integration_grid: Float[Tensor, "grid"],  # noqa: F821
    noise_standard_deviation: float,
    generator: torch.Generator,
    context_actions: Float[Tensor, "batch context_time action"] | None = None,
) -> tuple[Float[Tensor, "batch"], Float[Tensor, "batch"]]:  # noqa: F821
    batch_size = target_video.shape[0]
    video_state, audio_state = target_video.float(), target_audio.float()
    video_divergence = video_state.new_zeros(batch_size)
    audio_divergence = audio_state.new_zeros(batch_size)
    for grid_index in range(integration_grid.shape[0] - 1, 0, -1):
        time = integration_grid[grid_index].expand(batch_size)
        step = integration_grid[grid_index] - integration_grid[grid_index - 1]
        with torch.enable_grad():
            video_input = video_state.detach().requires_grad_()
            audio_input = audio_state.detach().requires_grad_()
            video_velocity, audio_velocity = predictor(
                video_input,
                audio_input,
                time,
                time,
                context_video,
                context_audio,
                actions,
                context_actions,
            )
            video_probe = rademacher(video_input.shape, generator, video_input.device)
            audio_probe = rademacher(audio_input.shape, generator, audio_input.device)
            video_product, audio_product = torch.autograd.grad(
                (video_velocity, audio_velocity), (video_input, audio_input), (video_probe, audio_probe)
            )
        video_divergence += step * (video_product * video_probe).flatten(1).sum(dim=1)
        audio_divergence += step * (audio_product * audio_probe).flatten(1).sum(dim=1)
        video_state = video_state - step * video_velocity.detach()
        audio_state = audio_state - step * audio_velocity.detach()
    return (
        gaussian_negative_log_likelihood(video_state, noise_standard_deviation) + video_divergence,
        gaussian_negative_log_likelihood(audio_state, noise_standard_deviation) + audio_divergence,
    )


def bits_per_dimension(negative_log_likelihood: Tensor, reference: Tensor) -> Tensor:
    dimensions = rearrange(reference, "batch ... -> batch (...)").shape[1]
    return negative_log_likelihood / (dimensions * math.log(2))
