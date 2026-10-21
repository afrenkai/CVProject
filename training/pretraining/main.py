import json
from pathlib import Path
import hydra
import torch
from omegaconf import DictConfig

from train_datasets.media import (
    load_av_clip,
    load_media_manifest,
    validate_pretraining_records,
)
from training.pretraining.checkpoint import save_pretraining_checkpoint
from training.pretraining.models import build_pretraining_models
from training.pretraining.normalization import fit_audio_visual_normalizer
from training.pretraining.stages import (
    pretrain_audio_encoder,
    pretrain_latent_world_model,
    pretrain_video_decoder,
)
from training.runtime import cuda_device
from transport.flow import AudioVisualFlowTransport


def summarize_losses(losses: list[float]) -> dict[str, float]:
    if not losses:
        raise ValueError("Cannot summarize an empty loss sequence")
    return {
        "initial": losses[0],
        "final": losses[-1],
        "minimum": min(losses),
    }


@hydra.main(config_path=None, version_base=None)
def main(config: DictConfig) -> None:
    device = cuda_device(config.runtime.seed)
    transport = AudioVisualFlowTransport.from_omegaconf(config.transport)
    total_steps = config.data.context_steps + config.data.target_steps
    records = load_media_manifest(Path(config.data.manifest), "train")
    excluded = list(config.data.excluded_datasets)
    validate_pretraining_records(records, excluded)
    av_clips = [
        load_av_clip(
            record,
            config.data.frames_per_second,
            total_steps,
            config.data.image_size,
            config.audio_encoder.sample_rate,
            config.audio_encoder.input_channels,
        )
        for record in records
    ]
    print(
        json.dumps(
            {
                "training_clips": len(records),
                "training_domains": sorted({record.dataset for record in records}),
                "excluded_games": sorted(excluded),
            }
        ),
        flush=True,
    )

    models = build_pretraining_models(config, device)

    waveforms = [clip.waveform[None].to(device) for clip in av_clips]
    audio_losses = pretrain_audio_encoder(
        models.audio_encoder,
        models.audio_decoder,
        waveforms,
        config.audio_pretraining.iterations,
        config.audio_pretraining.learning_rate,
    )
    models.audio_encoder.requires_grad_(False)
    models.audio_encoder.eval()
    if models.audio_encoder.total_stride * config.data.frames_per_second != config.audio_encoder.sample_rate:
        raise ValueError("Causal audio tokens must align with the video frame rate")
    raw_latent_examples: list[tuple[torch.Tensor, torch.Tensor]] = []
    with torch.no_grad():
        for clip in av_clips:
            video_latents = models.video_encoder(clip.frames[None].to(device))
            audio_latents = models.audio_encoder(clip.waveform[None].to(device))[0]
            raw_latent_examples.append((video_latents, audio_latents))

    video_examples = [
        (
            raw_latent_examples[example_index][0],
            av_clips[example_index].frames[None].to(device),
        )
        for example_index in range(len(av_clips))
    ]
    video_decoder_losses = pretrain_video_decoder(
        models.video_decoder,
        video_examples,
        config.video_decoder_pretraining.iterations,
        config.video_decoder_pretraining.learning_rate,
    )
    models.normalizer = fit_audio_visual_normalizer(
        raw_latent_examples,
        config.normalization.minimum_standard_deviation,
    ).to(device)
    latent_examples = [
        (
            models.normalizer.normalize_video(video_latents),
            models.normalizer.normalize_audio(audio_latents),
        )
        for video_latents, audio_latents in raw_latent_examples
    ]

    def save_world_checkpoint(iteration: int) -> None:
        checkpoint_path = save_pretraining_checkpoint(Path(config.output.directory), models, config)
        print(json.dumps({"world_step": iteration, "checkpoint": str(checkpoint_path)}), flush=True)

    world_losses, slim_lsd_losses, final_code_ids = pretrain_latent_world_model(
        transport,
        models.predictor,
        models.latent_action_model,
        latent_examples,
        config.data.context_steps,
        config.world_pretraining.iterations,
        config.world_pretraining.learning_rate,
        config.latent_actions.commitment_weight,
        config.slim_lsd.start_iteration,
        config.slim_lsd.weight,
        config.slim_lsd.maximum_time_gap,
        save_world_checkpoint,
        config.world_pretraining.checkpoint_interval,
    )
    checkpoint_path = save_pretraining_checkpoint(Path(config.output.directory), models, config)
    print(
        json.dumps(
            {
                "audio_visual_sources": [record.model_dump(mode="json") for record in records],
                "audio_loss": summarize_losses(audio_losses),
                "video_decoder_loss": summarize_losses(video_decoder_losses),
                "world_loss": summarize_losses(world_losses),
                "slim_lsd_loss": summarize_losses(slim_lsd_losses),
                "latent_action_codes": final_code_ids[0].tolist(),
                "checkpoint": str(checkpoint_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
