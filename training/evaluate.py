import argparse
import os
from pathlib import Path
import torch
import wandb
from matplotlib import pyplot
from torch.utils.data import DataLoader, Dataset, default_collate
from data.control_types import ControlLayout
from data.windows import build_validation_datasets
from models.joint_encoder import JointEncoder, build_encoder_config
from plotting.samples import sample_report, spread_figure
from training.objective import WorldModelObjective, checkpoint_weights, restore_objective
from training.samples import ensemble_errors, ensemble_metrics, sweep_rollouts
from training.training_types import PretrainConfig


def flow_losses(
    objectives: dict[str, WorldModelObjective],
    encoder: JointEncoder,
    dataset: Dataset,
    name: str,
    batch_size: int,
    workers: int,
    device: torch.device,
) -> dict[str, float]:
    parts = ("video_flow_loss", "audio_flow_loss")
    totals = {f"{label}/{name}/{part}": 0.0 for label in objectives for part in parts}
    batches = 0
    for index, batch in enumerate(DataLoader(dataset, batch_size=batch_size, num_workers=workers)):
        batch = {key: tensor.to(device) for key, tensor in batch.items()}
        video, audio = encoder.encode(batch["frames"], batch["waveform"])
        for label, objective in objectives.items():
            with torch.random.fork_rng(devices=[device.index or 0]):
                torch.manual_seed(index)
                outputs = objective.flow_loss(
                    objective.normalizer.normalize_video(video.float()),
                    objective.normalizer.normalize_audio(audio.float()),
                    batch["actions"],
                    batch["action_present"],
                )
            for part in parts:
                totals[f"{label}/{name}/{part}"] += outputs[part].item()
        batches = index + 1
    return {key: value / max(batches, 1) for key, value in totals.items()}


@torch.no_grad()
def main(
    run: Path, checkpoint: Path, windows: int, batch_size: int, workers: int, spread_windows: int, members: int
) -> None:
    config = PretrainConfig.model_validate_json((run / "pretrain_config.json").read_text())
    config = config.model_copy(update={"data": config.data.model_copy(update={"validation_windows": windows})})
    device = torch.device("cuda")
    step = int(checkpoint.name.removeprefix("checkpoint-"))
    bfloat16 = config.trainer.precision == "bf16"
    encoder_config = build_encoder_config(config.encoder, bfloat16=bfloat16)
    encoder = JointEncoder(encoder_config, device).eval()
    objectives = {
        label: restore_objective(config, encoder_config, checkpoint_weights(checkpoint, label == "ema"), device)
        for label in ("raw", "ema")
    }
    validation = build_validation_datasets(config.data, encoder_config, config.encoder.total_frames, ControlLayout())
    metrics: dict[str, float] = {"checkpoint_step": step}
    for name, dataset in validation.items():
        losses = flow_losses(objectives, encoder, dataset, name, batch_size, workers, device)
        metrics |= losses
        print(f"step {step} {name}: " + "  ".join(f"{key} {value:.4f}" for key, value in losses.items()), flush=True)

    shown = validation.get("d2e") or next(iter(validation.values()))
    samples = {key: tensor.to(device) for key, tensor in default_collate([shown[index] for index in range(4)]).items()}
    figures = {}
    for label, objective in objectives.items():
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bfloat16):
            results = sweep_rollouts(objective, encoder, samples, (1, 4, 8, 16), seed=0)
        report, plots = sample_report(results, samples["frames"], samples.get("waveform"), encoder_config)
        metrics |= {key.replace("samples/", f"samples_{label}/"): value for key, value in report.items()}
        figures |= {f"samples_{label}/{plot}": figure for plot, figure in plots.items()}

    by_dataset = {}
    for name, dataset in validation.items():
        count = min(spread_windows, len(dataset))
        chunks = []
        for start in range(0, count, 4):
            chunk = default_collate([dataset[index] for index in range(start, min(start + 4, count))])
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bfloat16):
                chunks.append(
                    ensemble_errors(
                        objectives["ema"],
                        encoder,
                        {key: tensor.to(device) for key, tensor in chunk.items()},
                        members,
                        16,
                        seed=start,
                    )
                )
        by_dataset[name] = {key: torch.stack([chunk[key] for chunk in chunks]).mean(dim=0) for key in chunks[0]}
        metrics |= ensemble_metrics(by_dataset[name], f"ensemble_ema/{name}")
    seconds_per_step = encoder_config.video_frames_per_step / encoder_config.video_rate
    figures["ensemble_ema/spread_skill_by_dataset"] = spread_figure(by_dataset, seconds_per_step)
    print(
        "ensemble " + "  ".join(f"{key} {value:.3f}" for key, value in metrics.items() if "ensemble_ema/" in key),
        flush=True,
    )

    wandb.init(
        project=config.trainer.wandb_project,
        id=f"{run.name}-eval",
        name=f"{run.name}-eval",
        resume="allow",
        dir=str(run),
        config=config.model_dump(mode="json"),
    )
    wandb.define_metric("checkpoint_step")
    wandb.define_metric("*", step_metric="checkpoint_step")
    wandb.log(metrics | {key: wandb.Image(figure) for key, figure in figures.items()})
    wandb.finish()
    for figure in figures.values():
        pyplot.close(figure)
    (checkpoint / "evaluated").touch()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--windows", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--spread-windows", type=int, default=16)
    parser.add_argument("--members", type=int, default=8)
    parser.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")) - 2)
    command = parser.parse_args()
    main(
        command.run,
        command.checkpoint,
        command.windows,
        command.batch_size,
        command.workers,
        command.spread_windows,
        command.members,
    )
