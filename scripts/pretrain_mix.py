import time
from collections import Counter
from pathlib import Path
import fire
import torch
from torch.autograd.profiler import emit_nvtx
from torch.profiler import ProfilerActivity, profile, record_function, schedule
from torch.utils.data import DataLoader, Subset
from architecture.pretraining.joint_encoder import JointEncoder
from architecture.pretraining.model_types import AudioTokenConfig, TransformerStageConfig, WorldModelConfig
from architecture.pretraining.normalization import AudioVisualNormalizer
from train_datasets.control_types import ControlLayout
from train_datasets.latent_cache import build_encoder_config
from train_datasets.pool_windows import PoolWindowDataset, load_pool_records
from training.simple_objective import build_latent_objective, feature_moments
from transport.transport_types import TransportConfig


def main(
    catalog: str,
    output: str,
    mode: str,
    windows: int = 40000,
    worker_refresh_windows: int = 4096,
    epochs: int = 2,
    batch_size: int = 32,
    learning_rate: float = 3e-4,
    normalizer_windows: int = 1024,
    workers: int = 12,
    seed: int = 0,
    log_every: int = 50,
    hidden_dimension: int = 512,
    heads: int = 8,
    layers: int = 4,
    head_hidden_dimension: int = 256,
    head_heads: int = 8,
    head_layers: int = 2,
    prediction: str = "data",
    video_target: str = "change",
    device: str = "cuda",
    precision: str = "fp32",
    max_steps: int = 0,
    profile_steps: int = 0,
    profile_warmup_steps: int = 10,
    cuda_profiler: bool = False,
) -> None:
    assert mode in {"audio_visual", "video_only"}
    assert precision in {"fp32", "bf16"}
    output_directory = Path(output)
    output_directory.mkdir(parents=True, exist_ok=True)
    encoder_cfg = build_encoder_config(
        context_frames=12, audio_model="mispeech/dasheng-base", bfloat16=precision == "bf16"
    )
    layout = ControlLayout()
    records = load_pool_records(Path(catalog), "train")
    print(f"Training records: {len(records)} {dict(Counter(record.dataset for record in records))}", flush=True)
    dataset = PoolWindowDataset(
        records,
        windows,
        seed,
        total_frames=42,
        video_frames_per_step=encoder_cfg.video_frames_per_step,
        action_dimension=layout.dimension,
        image_size=encoder_cfg.image_size,
        video_rate=encoder_cfg.video_rate,
        sample_rate=encoder_cfg.sample_rate,
    )
    encoder = JointEncoder(encoder_cfg, torch.device(device))

    def batches(source: Subset | PoolWindowDataset) -> DataLoader:
        return DataLoader(source, batch_size=batch_size, num_workers=workers, prefetch_factor=4 if workers else None)

    fitted_video, fitted_audio = [], []
    for batch in batches(Subset(dataset, range(normalizer_windows))):
        with torch.no_grad():
            video, audio = encoder.encode(batch["frames"].to(device), batch["waveform"].to(device))
        fitted_video.append(video.half())
        fitted_audio.append(audio.half())
    video_mean, video_std = feature_moments(torch.cat(fitted_video), (0, 1, 2))
    audio_mean, audio_std = feature_moments(torch.cat(fitted_audio), (0, 1))
    normalizer = AudioVisualNormalizer(video_mean.cpu(), video_std.cpu(), audio_mean.cpu(), audio_std.cpu()).to(device)
    del fitted_video, fitted_audio

    world_model = WorldModelConfig(
        action_dimension=64,
        backbone_attention="quadratic",
        backbone=TransformerStageConfig(hidden_dimension=hidden_dimension, heads=heads, layers=layers),
        diffusion_head=TransformerStageConfig(
            hidden_dimension=head_hidden_dimension, heads=head_heads, layers=head_layers
        ),
        prediction=prediction,
        audio_tokens=AudioTokenConfig(channels=2, frames_per_step=5),
    )
    transport = TransportConfig(video_target=video_target)
    torch.manual_seed(seed)
    objective = build_latent_objective(
        encoder_cfg,
        world_model,
        transport,
        normalizer.cpu(),
        1.0 if mode == "audio_visual" else 0.0,
        output_directory,
        torch.device(device),
    )
    normalizer = normalizer.to(device)
    optimizer = torch.optim.AdamW(objective.parameters(), lr=learning_rate)
    objective.train()
    torch.manual_seed(seed + 1)
    step = 0
    started = time.perf_counter()
    running = {"video_flow_loss": 0.0, "audio_flow_loss": 0.0}
    profiling = profile_steps > 0
    profiler = (
        profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(wait=profile_warmup_steps, warmup=2, active=profile_steps, repeat=1),
            record_shapes=True,
            with_stack=False,
        )
        if profiling
        else None
    )
    last_step = profile_warmup_steps + 2 + profile_steps if profiling else max_steps
    autocast = torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision == "bf16")
    if profiler is not None:
        profiler.start()
    if cuda_profiler:
        torch.cuda.profiler.start()
    with emit_nvtx(enabled=cuda_profiler):
        for epoch in range(epochs):
            chunks = (
                Subset(dataset, range(chunk_start, min(chunk_start + worker_refresh_windows, windows)))
                for chunk_start in range(0, windows, worker_refresh_windows)
            )
            for batch in (batch for chunk in chunks for batch in batches(chunk)):
                with record_function("encode"), torch.no_grad(), autocast:
                    video, audio = encoder.encode(batch["frames"].to(device), batch["waveform"].to(device))
                    video = normalizer.normalize_video(video.float())
                    audio = normalizer.normalize_audio(audio.float())
                    if mode == "video_only":
                        audio = torch.zeros_like(audio)
                with record_function("train_step"):
                    with autocast:
                        outputs = objective.feature_loss(
                            video,
                            audio,
                            batch["actions"].to(device),
                            batch["action_present"].to(device),
                            batch["example_seed"].to(device),
                        )
                    optimizer.zero_grad()
                    outputs["loss"].backward()
                    optimizer.step()
                step += 1
                if profiler is not None:
                    profiler.step()
                if last_step and step >= last_step:
                    break
                for name in running:
                    running[name] += outputs[name].mean().item()
                if step % log_every == 0:
                    losses = "  ".join(f"{name} {value / log_every:.4f}" for name, value in running.items())
                    elapsed = time.perf_counter() - started
                    print(
                        f"[{mode}] epoch {epoch} step {step}: {losses}  ({step * batch_size / elapsed:.1f} windows/s)",
                        flush=True,
                    )
                    running = dict.fromkeys(running, 0.0)
            if last_step and step >= last_step:
                break
            checkpoint = output_directory / f"checkpoint_epoch{epoch + 1}.pt"
            torch.save(
                {
                    "objective": objective.state_dict(),
                    "normalizer": normalizer.cpu().state_dict(),
                    "world_model": world_model.model_dump(),
                    "transport": transport.model_dump(),
                    "encoder": encoder_cfg.model_dump(mode="json"),
                    "mode": mode,
                    "windows": windows,
                    "epochs_completed": epoch + 1,
                    "steps": step,
                },
                checkpoint,
            )
            normalizer = normalizer.to(device)
            print(f"Saved {checkpoint}", flush=True)
    if cuda_profiler:
        torch.cuda.profiler.stop()
    if profiler is not None:
        profiler.stop()
        trace = output_directory / f"profile_{precision}.json"
        profiler.export_chrome_trace(str(trace))
        print(profiler.key_averages().table(sort_by="cpu_time_total", row_limit=25), flush=True)
        print(profiler.key_averages().table(sort_by="self_cuda_time_total", row_limit=40), flush=True)
        print(f"Saved {trace}", flush=True)


if __name__ == "__main__":
    fire.Fire(main)
