import atexit
import shutil
import tempfile
from pathlib import Path
import fire
import matplotlib
import torch
import yaml
from datasets import load_dataset
from huggingface_hub import snapshot_download
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig
from architecture.pretraining.encoder_types import FrozenEncoderConfig
from architecture.pretraining.joint_encoder import JointEncoder
from architecture.pretraining.model_types import TransformerStageConfig, WorldModelConfig
from architecture.pretraining.normalization import AudioVisualNormalizer
from evaluation.evaluation_types import ForecastSummary
from evaluation.forecast import forecast_window
from train_datasets.control_types import ControlLayout
from train_datasets.simple_av import AVClip, HFAVDataset, SimpleAVDataset, scan_hf_windows, scan_video_windows
from training.simple_loop import eval_epoch, train_epoch
from training.simple_objective import SimpleObjectiveConfig, build_simple_objective
from transport.transport_types import ForecastConfig

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def resolve_audio_model(audio_model: str) -> Path:
    p = Path(audio_model)
    if p.exists():
        return p
    return Path(snapshot_download(audio_model))


def main(
    audio_model: str = "mispeech/dasheng-base",
    video: str = "",
    hf_dataset: str = "",
    hf_split: str = "test[:30]",
    output: str = "demo.png",
) -> None:
    assert video or hf_dataset, "Provide --video or --hf_dataset"

    device = torch.device("cpu")

    video_rate = 30
    total_frames = 30
    tmp_path = Path(tempfile.mkdtemp(prefix="speakup_demo_"))
    atexit.register(shutil.rmtree, tmp_path, ignore_errors=True)

    if video:
        windows = scan_video_windows(Path(video), total_frames, video_rate, stride_seconds=0.5)
    else:
        hf_ds = load_dataset(hf_dataset, split=hf_split)
        windows = scan_hf_windows(hf_ds, total_frames, video_rate, stride_seconds=0.5)

    assert windows, "No clips found — videos may be too short"
    print(f"Found {len(windows)} clips")

    audio_path = resolve_audio_model(audio_model)
    audio_feature_dim: int = AutoConfig.from_pretrained(
        str(audio_path), trust_remote_code=True, local_files_only=True
    ).encoder_kwargs["embed_dim"]
    print(f"Audio model: {audio_path.name}  embed_dim={audio_feature_dim}")

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

    world_model_cfg = WorldModelConfig(
        action_dimension=64,
        backbone_attention="quadratic",
        backbone=TransformerStageConfig(hidden_dimension=128, heads=4, layers=2),
        diffusion_head=TransformerStageConfig(hidden_dimension=64, heads=4, layers=1),
    )

    video_dim = encoder_cfg.video_latent_dimension
    audio_dim = encoder_cfg.audio_latent_dimension
    dummy_normalizer = AudioVisualNormalizer(
        torch.zeros(video_dim),
        torch.ones(video_dim),
        torch.zeros(audio_dim),
        torch.ones(audio_dim),
    )
    normalizer_path = tmp_path / "normalizer.pt"
    torch.save(dummy_normalizer.state_dict(), normalizer_path)

    layout = ControlLayout()
    scales_path = tmp_path / "action_scales.yaml"
    scales_path.write_text(
        yaml.dump(
            {
                "fit_split": "train",
                "fit_windows": 0,
                "axis_scales": [1.0] * len(layout.axis_names),
            }
        )
    )

    obj_cfg = SimpleObjectiveConfig(
        normalizer=normalizer_path,
        action_scales=scales_path,
        encoder=encoder_cfg,
        world_model=world_model_cfg,
        context_frames=encoder_cfg.context_frames,
        video_frames_per_step=encoder_cfg.video_frames_per_step,
        control_hidden_dimension=128,
        control_layout=layout,
    )

    encoder = JointEncoder(encoder_cfg, device)
    objective = build_simple_objective(obj_cfg, encoder.encode, device)
    print(f"Predictor params: {sum(p.numel() for p in objective.predictor.parameters()):,}")

    split = max(1, len(windows) // 5)
    train_windows, eval_windows = windows[split:], windows[:split]

    def make_dataset(split_windows: list[AVClip] | list[tuple[int, float]]) -> Dataset:
        if video:
            return SimpleAVDataset(
                split_windows,
                total_frames,
                encoder_cfg.video_frames_per_step,
                layout.dimension,
                encoder_cfg.image_size,
                encoder_cfg.video_rate,
                encoder_cfg.sample_rate,
            )
        return HFAVDataset(
            hf_ds,
            split_windows,
            total_frames,
            encoder_cfg.video_frames_per_step,
            layout.dimension,
            encoder_cfg.image_size,
            encoder_cfg.video_rate,
        )

    train_dl = DataLoader(make_dataset(train_windows), batch_size=1, shuffle=True, num_workers=0)
    eval_dl = DataLoader(make_dataset(eval_windows), batch_size=1, shuffle=False, num_workers=0)

    optimizer = torch.optim.AdamW(objective.parameters(), lr=1e-4)

    print("Training...")
    print(f"  {train_epoch(objective, train_dl, optimizer, device)}")
    print("Eval...")
    print(f"  {eval_epoch(objective, eval_dl, device)}")

    objective.eval()
    forecast_config = ForecastConfig()
    summaries: list[ForecastSummary] = []
    reviewed: list[tuple[Tensor, Tensor]] = []
    with torch.no_grad():
        for batch in eval_dl:
            video_latents, audio_latents = encoder.encode(batch["frames"].to(device), batch["waveform"].to(device))
            video_latents = objective.normalizer.normalize_video(video_latents)
            audio_latents = objective.normalizer.normalize_audio(audio_latents)
            if not reviewed:
                true_video = video_latents[:, objective.context_steps :]
                true_audio = audio_latents[:, objective.context_steps :]
            summaries.append(
                forecast_window(
                    objective,
                    video_latents,
                    audio_latents,
                    batch["actions"].to(device),
                    batch["action_present"].to(device),
                    int(batch["example_seed"].item()),
                    forecast_config,
                    seed=0,
                    review=(
                        None if reviewed else lambda video_mean, audio_mean: reviewed.append((video_mean, audio_mean))
                    ),
                )
            )
    pred_video, pred_audio = reviewed[0]

    print(f"Forecast on {len(summaries)} eval clips ({forecast_config.n_futures} futures each), normalized-latent MSE:")
    modality_statistics = {
        "video": [summary.video for summary in summaries],
        "audio": [summary.audio for summary in summaries],
    }
    for modality, statistics in modality_statistics.items():
        sample_mse = sum(entry.mean_sample_mse for entry in statistics) / len(statistics)
        mean_mse = sum(entry.sample_mean_mse for entry in statistics) / len(statistics)
        persistence_mse = sum(entry.persistence_mse for entry in statistics) / len(statistics)
        print(
            f"  {modality}: sample {sample_mse:.4f}  mean-of-samples {mean_mse:.4f}  "
            f"persistence {persistence_mse:.4f}"
        )

    n_feat = 32
    true_v = true_video[0].mean(dim=1)[:, :n_feat].cpu().numpy()
    pred_v = pred_video[0].mean(dim=1)[:, :n_feat].cpu().numpy()
    true_a = true_audio[0][:, :n_feat].cpu().numpy()
    pred_a = pred_audio[0][:, :n_feat].cpu().numpy()

    vmin, vmax = min(true_v.min(), pred_v.min()), max(true_v.max(), pred_v.max())
    amin, amax = min(true_a.min(), pred_a.min()), max(true_a.max(), pred_a.max())

    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].imshow(true_v, aspect="auto", vmin=vmin, vmax=vmax)
    axes[0, 0].set_title("True video latent future")
    axes[0, 1].imshow(pred_v, aspect="auto", vmin=vmin, vmax=vmax)
    axes[0, 1].set_title("Predicted video latent future (mean of 8 samples)")
    axes[1, 0].imshow(true_a, aspect="auto", vmin=amin, vmax=amax)
    axes[1, 0].set_title("True audio latent future")
    axes[1, 1].imshow(pred_a, aspect="auto", vmin=amin, vmax=amax)
    axes[1, 1].set_title("Predicted audio latent future (mean of 8 samples)")

    for ax in axes.flat:
        ax.set_xlabel("feature[:32]")
        ax.set_ylabel("time step")

    figure.tight_layout()
    figure.savefig(output, dpi=150)
    print(f"Saved {output}")


if __name__ == "__main__":
    fire.Fire(main)
