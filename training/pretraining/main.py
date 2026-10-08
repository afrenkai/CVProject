import json
from pathlib import Path
import hydra
import torch
from omegaconf import DictConfig
from train_datasets.huggingface import load_audio_clip, load_av_clip
from train_datasets.manifest import load_episode_manifest
from train_datasets.media import load_manifest_audio_clip, load_manifest_av_clip
from training.checkpoint import save_checkpoint
from training.pretraining.loops import (
    pretrain_audio_encoder,
    pretrain_latent_world_model,
    pretrain_video_decoder,
)
from training.pretraining.models import build_pretraining_models
from training.pretraining.normalization import fit_audio_visual_normalizer
from transport.flow import AudioVisualFlowTransport


def summarize_losses(losses: list[float]) -> dict[str, float]:
    if not losses:
        raise ValueError("Cannot summarize an empty loss sequence")
    return {
        "initial": losses[0],
        "final": losses[-1],
        "minimum": min(losses),
    }


@hydra.main(config_path="../../configs", config_name="pretraining", version_base=None)
def main(config: DictConfig) -> None:
    torch.manual_seed(config.runtime.seed)
    device = torch.device(config.runtime.device)
    transport = AudioVisualFlowTransport.from_omegaconf(config.transport)
    total_steps = config.data.context_steps + config.data.target_steps
    sample_count = total_steps * config.audio_encoder.sample_rate // config.data.frames_per_second

    av_clips = []
    for source in config.sources.audio_visual:
        arrow_path = None if source.arrow is None else Path(source.arrow)
        av_clips.append(
            load_av_clip(
                str(source.repository),
                str(source.split),
                arrow_path,
                int(source.sample_index),
                config.data.frames_per_second,
                total_steps,
                config.data.image_size,
                config.audio_encoder.sample_rate,
                config.audio_encoder.input_channels,
            )
        )
    for source in config.sources.audio_visual_manifests:
        records = load_episode_manifest(Path(source.path))
        record = records[int(source.episode_index)]
        if record.split != "train":
            raise ValueError(f"Pretraining manifest episode must use the train split: {record.episode_id}")
        av_clips.append(
            load_manifest_av_clip(
                record,
                config.data.frames_per_second,
                total_steps,
                config.data.image_size,
                config.audio_encoder.sample_rate,
                config.audio_encoder.input_channels,
            )
        )
    audio_clips = [
        load_audio_clip(
            str(source.repository),
            str(source.split),
            int(source.sample_index),
            str(source.audio_column),
            config.audio_encoder.sample_rate,
            config.audio_encoder.input_channels,
            sample_count,
        )
        for source in config.sources.audio
    ]
    for source in config.sources.audio_manifests:
        records = load_episode_manifest(Path(source.path))
        record = records[int(source.episode_index)]
        if record.split != "train":
            raise ValueError(f"Pretraining manifest episode must use the train split: {record.episode_id}")
        audio_clips.append(
            load_manifest_audio_clip(
                record,
                config.audio_encoder.sample_rate,
                config.audio_encoder.input_channels,
                sample_count,
            )
        )
    if not av_clips:
        raise ValueError("Generic world-model pretraining requires paired AV clips")

    models = build_pretraining_models(config, device)

    waveforms = [clip.waveform[None].to(device) for clip in audio_clips]
    waveforms.extend(clip.waveform[None].to(device) for clip in av_clips)
    audio_losses = pretrain_audio_encoder(
        models.audio_encoder,
        models.audio_decoder,
        waveforms,
        config.audio_pretraining.iterations,
        config.audio_pretraining.learning_rate,
    )
    models.audio_encoder.requires_grad_(False)
    models.audio_encoder.eval()
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
    )
    checkpoint_path = save_checkpoint(
        Path(config.output.checkpoint),
        models.pretraining_checkpoint_modules(),
    )
    print(
        json.dumps(
            {
                "audio_sources": [{"dataset": clip.dataset, "sample_id": clip.sample_id} for clip in audio_clips],
                "audio_visual_sources": [{"dataset": clip.dataset, "sample_id": clip.sample_id} for clip in av_clips],
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
