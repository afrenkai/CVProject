import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor
from architecture.pretraining.audio.audio_decoder import CausalAudioDecoder


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
