import json
from collections.abc import Callable, Sequence
import torch
from beartype import beartype
from jaxtyping import Float, Int, jaxtyped
from torch import Tensor
from architecture.av_model import AudioVisualFlowPredictor
from architecture.pretraining.audio.audio_decoder import CausalAudioDecoder
from architecture.pretraining.audio.encoder import HybridAudioEncoder
from architecture.pretraining.latent_actions import LatentActionModel
from architecture.pretraining.video.video_decoder import DinoVideoDecoder
from losses.audio import audio_reconstruction_loss
from losses.av import flow_matching_loss, slim_lsd_loss
from losses.video import video_reconstruction_loss
from transport.flow import AudioVisualFlowTransport


@jaxtyped(typechecker=beartype)
def pretrain_video_decoder(
    decoder: DinoVideoDecoder,
    examples: Sequence[
        tuple[
            Float[Tensor, "batch time patch feature"],
            Float[Tensor, "batch time channel height width"],
        ]
    ],
    iterations: int,
    learning_rate: float,
) -> list[float]:
    if not examples or iterations <= 0:
        raise ValueError("Video-decoder pretraining requires examples and positive iterations")
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=learning_rate)
    losses: list[float] = []
    for iteration in range(iterations):
        latents, frames = examples[iteration % len(examples)]
        loss = video_reconstruction_loss(decoder, latents.detach(), frames)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if not torch.isfinite(loss):
            raise RuntimeError(f"Video-decoder pretraining failed on optimizer step {iteration + 1}")
        optimizer.step()
        losses.append(loss.item())
        if (iteration + 1) % 100 == 0:
            print(json.dumps({"stage": "video_decoder", "step": iteration + 1, "loss": loss.item()}), flush=True)
    return losses


@jaxtyped(typechecker=beartype)
def pretrain_audio_encoder(
    encoder: HybridAudioEncoder,
    decoder: CausalAudioDecoder,
    waveforms: Sequence[Float[Tensor, "batch channel sample"]],
    iterations: int,
    learning_rate: float,
) -> list[float]:
    if not waveforms or iterations <= 0:
        raise ValueError("Audio pretraining requires waveforms and positive iterations")
    optimizer = torch.optim.AdamW(
        [*encoder.parameters(), *decoder.parameters()],
        lr=learning_rate,
    )
    losses: list[float] = []
    for iteration in range(iterations):
        waveform = waveforms[iteration % len(waveforms)]
        latents = encoder(waveform)[0]
        loss = audio_reconstruction_loss(decoder, latents, waveform)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if not torch.isfinite(loss):
            raise RuntimeError(f"Audio pretraining failed on optimizer step {iteration + 1}")
        optimizer.step()
        losses.append(loss.item())
        if (iteration + 1) % 100 == 0:
            print(json.dumps({"stage": "audio_encoder", "step": iteration + 1, "loss": loss.item()}), flush=True)
    return losses


@jaxtyped(typechecker=beartype)
def pretrain_latent_world_model(
    transport: AudioVisualFlowTransport,
    predictor: AudioVisualFlowPredictor,
    latent_action_model: LatentActionModel,
    latent_examples: Sequence[
        tuple[
            Float[Tensor, "batch full_time patch video_feature"],
            Float[Tensor, "batch full_time audio_feature"],
        ]
    ],
    context_steps: int,
    iterations: int,
    learning_rate: float,
    commitment_weight: float,
    slim_lsd_start: int,
    slim_lsd_weight: float,
    maximum_time_gap: float,
    checkpoint_callback: Callable[[int], None] | None = None,
    checkpoint_interval: int = 1000,
) -> tuple[list[float], list[float], Int[Tensor, "batch target_time"]]:
    if not latent_examples or iterations <= 0 or checkpoint_interval <= 0:
        raise ValueError("World-model pretraining requires latent examples and positive iterations")
    # TODO: ablate optimizers, muon first
    optimizer = torch.optim.AdamW(
        [*predictor.parameters(), *latent_action_model.parameters()],
        lr=learning_rate,
    )
    total_losses: list[float] = []
    lsd_losses: list[float] = []
    final_code_ids: Int[Tensor, "batch target_time"] | None = None
    for iteration in range(iterations):
        video_latents, audio_latents = latent_examples[iteration % len(latent_examples)]
        previous_video = video_latents[:, context_steps - 1 : -1]
        next_video = video_latents[:, context_steps:]
        previous_audio = audio_latents[:, context_steps - 1 : -1]
        next_audio = audio_latents[:, context_steps:]
        latent_actions, code_ids, encoded_actions, commitment_loss = latent_action_model(
            previous_video,
            next_video,
            previous_audio,
            next_audio,
        )
        flow_loss = flow_matching_loss(
            transport,
            predictor,
            video_latents,
            audio_latents,
            context_steps,
            latent_actions,
        )
        loss = flow_loss + commitment_weight * commitment_loss
        lsd_loss = torch.zeros((), device=loss.device)
        if iteration >= slim_lsd_start:
            lsd_loss = slim_lsd_loss(
                transport,
                predictor,
                video_latents,
                audio_latents,
                context_steps,
                latent_actions,
                maximum_time_gap,
            )
            loss = loss + slim_lsd_weight * lsd_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if not torch.isfinite(loss):
            raise RuntimeError(f"World-model pretraining failed on optimizer step {iteration + 1}")
        optimizer.step()
        latent_action_model.update_codebook(encoded_actions.detach(), code_ids)
        total_losses.append(loss.item())
        lsd_losses.append(lsd_loss.item())
        final_code_ids = code_ids.detach()
        if (iteration + 1) % 100 == 0:
            print(json.dumps({"stage": "world_model", "step": iteration + 1, "loss": loss.item()}), flush=True)
        if checkpoint_callback is not None and (iteration + 1) % checkpoint_interval == 0:
            checkpoint_callback(iteration + 1)
    if final_code_ids is None:
        raise RuntimeError("World-model pretraining produced no latent actions")
    return total_losses, lsd_losses, final_code_ids
