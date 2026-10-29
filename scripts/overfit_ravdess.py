import atexit
import shutil
import tempfile
from pathlib import Path
import fire
import torch
import yaml
from datasets import load_dataset
from huggingface_hub import snapshot_download
from torch import Tensor
from transformers import AutoConfig
from architecture.pretraining.encoder_types import FrozenEncoderConfig
from architecture.pretraining.joint_encoder import JointEncoder
from architecture.pretraining.model_types import TransformerStageConfig, WorldModelConfig
from architecture.pretraining.normalization import AudioVisualNormalizer
from evaluation.evaluation_types import ForecastStatistics
from evaluation.forecast import forecast_window
from train_datasets.control_types import ControlLayout
from train_datasets.simple_av import HFAVDataset, scan_hf_windows
from training.simple_objective import SimpleObjectiveConfig, build_simple_objective
from transport.transport_types import ForecastConfig


def summarize(statistics: list[ForecastStatistics]) -> str:
    sample_mse = sum(entry.mean_sample_mse for entry in statistics) / len(statistics)
    mean_mse = sum(entry.sample_mean_mse for entry in statistics) / len(statistics)
    persistence_mse = sum(entry.persistence_mse for entry in statistics) / len(statistics)
    return f"sample {sample_mse:.4f}  mean-of-samples {mean_mse:.4f}  persistence {persistence_mse:.4f}"


def main(
    hf_dataset: str = "mteb/RAVDESS_AV",
    hf_split: str = "test[:8]",
    audio_model: str = "mispeech/dasheng-base",
    steps: int = 2000,
    batch_size: int = 16,
    learning_rate: float = 3e-4,
    minimum_rms_fraction: float = 0.25,
    log_every: int = 100,
) -> None:
    device = torch.device("cpu")
    video_rate = 30
    total_frames = 30
    tmp_path = Path(tempfile.mkdtemp(prefix="speakup_overfit_"))
    atexit.register(shutil.rmtree, tmp_path, ignore_errors=True)

    audio_path = Path(snapshot_download(audio_model))
    audio_feature_dim: int = AutoConfig.from_pretrained(
        str(audio_path), trust_remote_code=True, local_files_only=True
    ).encoder_kwargs["embed_dim"]
    encoder_cfg = FrozenEncoderConfig(
        video_model="vit_base_patch16_dinov3.lvd1689m",
        video_feature_dimension=768,
        video_layers=(3, 6, 9, 12),
        image_size=112,
        video_rate=video_rate,
        video_frames_per_step=3,
        sample_rate=16000,
        context_frames=6,
        audio_model_directory=audio_path,
        audio_feature_dimension=audio_feature_dim,
        audio_native_frame_rate=25,
        audio_frame_repeat=2,
        audio_tokens_per_step=5,
    )
    layout = ControlLayout()

    hf_ds = load_dataset(hf_dataset, split=hf_split)
    windows = scan_hf_windows(hf_ds, total_frames, video_rate, stride_seconds=0.5)
    dataset = HFAVDataset(
        hf_ds,
        windows,
        total_frames,
        encoder_cfg.video_frames_per_step,
        layout.dimension,
        encoder_cfg.image_size,
        video_rate,
    )
    items = [dataset[index] for index in range(len(dataset))]
    context_samples = encoder_cfg.context_frames * encoder_cfg.sample_rate // video_rate
    target_rms = torch.stack([item["waveform"][:, context_samples:].square().mean().sqrt() for item in items])
    keep = target_rms >= minimum_rms_fraction * target_rms.max()
    items = [items[index] for index in keep.nonzero().flatten().tolist()]
    print(f"{len(windows)} windows, {len(items)} with target audio RMS >= {minimum_rms_fraction:g} x max")

    encoder = JointEncoder(encoder_cfg, device)
    encoded = [encoder.encode(item["frames"][None], item["waveform"][None]) for item in items]
    video = torch.cat([video_latents for video_latents, audio_latents in encoded])
    audio = torch.cat([audio_latents for video_latents, audio_latents in encoded])
    actions = torch.stack([item["actions"] for item in items])
    action_present = torch.stack([item["action_present"] for item in items])
    example_seed = torch.arange(len(items))
    print(f"Cached latents: video {tuple(video.shape)}, audio {tuple(audio.shape)}")

    normalizer = AudioVisualNormalizer(
        video.mean(dim=(0, 1, 2)),
        video.std(dim=(0, 1, 2)).clamp_min(1e-4),
        audio.mean(dim=(0, 1)),
        audio.std(dim=(0, 1)).clamp_min(1e-4),
    )
    normalizer_path = tmp_path / "normalizer.pt"
    torch.save(normalizer.state_dict(), normalizer_path)
    scales_path = tmp_path / "action_scales.yaml"
    scales_path.write_text(
        yaml.dump({"fit_split": "train", "fit_windows": 0, "axis_scales": [1.0] * len(layout.axis_names)})
    )
    objective = build_simple_objective(
        SimpleObjectiveConfig(
            normalizer=normalizer_path,
            action_scales=scales_path,
            encoder=encoder_cfg,
            world_model=WorldModelConfig(
                action_dimension=64,
                backbone_attention="quadratic",
                backbone=TransformerStageConfig(hidden_dimension=128, heads=4, layers=2),
                diffusion_head=TransformerStageConfig(hidden_dimension=64, heads=4, layers=1),
            ),
            context_frames=encoder_cfg.context_frames,
            video_frames_per_step=encoder_cfg.video_frames_per_step,
            control_hidden_dimension=128,
            control_layout=layout,
        ),
        encoder.encode,
        device,
    )
    video = objective.normalizer.normalize_video(video)
    audio = objective.normalizer.normalize_audio(audio)

    optimizer = torch.optim.AdamW(objective.parameters(), lr=learning_rate)
    objective.train()
    running: dict[str, float] = {}
    for step in range(1, steps + 1):
        batch_index = torch.randint(len(items), (min(batch_size, len(items)),))
        outputs = objective.feature_loss(
            video[batch_index],
            audio[batch_index],
            actions[batch_index],
            action_present[batch_index],
            example_seed[batch_index],
        )
        optimizer.zero_grad()
        outputs["loss"].backward()
        optimizer.step()
        for name in ("video_flow_loss", "audio_flow_loss"):
            running[name] = running.get(name, 0.0) + outputs[name].mean().item()
        if step % log_every == 0:
            print(f"step {step}: " + "  ".join(f"{name} {value / log_every:.4f}" for name, value in running.items()))
            running = {}

    objective.eval()
    forecast_config = ForecastConfig()
    context = objective.context_steps

    def forecast(video_input: Tensor, audio_input: Tensor, index: int) -> tuple[ForecastStatistics, ForecastStatistics]:
        summary = forecast_window(
            objective,
            video_input[None],
            audio_input[None],
            actions[index : index + 1],
            action_present[index : index + 1],
            index,
            forecast_config,
            seed=0,
        )
        return summary.video, summary.audio

    matched = [forecast(video[index], audio[index], index) for index in range(len(items))]
    swapped_audio = [
        forecast(video[index], torch.cat((audio[(index + 1) % len(items), :context], audio[index, context:])), index)
        for index in range(len(items))
    ]
    swapped_video = [
        forecast(torch.cat((video[(index + 1) % len(items), :context], video[index, context:])), audio[index], index)
        for index in range(len(items))
    ]
    print(f"Forecast on the {len(items)} training windows, normalized-latent MSE:")
    print(f"  video | matched context:         {summarize([video_stats for video_stats, audio_stats in matched])}")
    print(
        f"  video | audio context swapped:   {summarize([video_stats for video_stats, audio_stats in swapped_audio])}"
    )
    print(f"  audio | matched context:         {summarize([audio_stats for video_stats, audio_stats in matched])}")
    print(
        f"  audio | video context swapped:   {summarize([audio_stats for video_stats, audio_stats in swapped_video])}"
    )


if __name__ == "__main__":
    fire.Fire(main)
