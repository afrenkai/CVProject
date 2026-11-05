import argparse
import os
from collections import Counter
from pathlib import Path
import torch
import torch.distributed as dist
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader, Dataset, Subset, default_collate
from transformers import TrainingArguments
from transformers.trainer_utils import get_last_checkpoint
from configs.loading import load_run_settings
from data.control_types import ControlLayout
from data.latent_cache import CachedWindowDataset
from data.windows import build_dataset, build_validation_datasets
from models.compilation import compile_blocks, set_recompile_limit
from models.encoder_types import FrozenEncoderConfig
from models.joint_encoder import JointEncoder, build_encoder_config
from models.normalization import AudioVisualNormalizer, RunningMoments
from training.objective import WorldModelObjective, build_objective
from training.optimization import build_optimizer, build_schedule, warmup_ema_update
from training.trainer import (
    EmaCallback,
    SampleCallback,
    SlimLsdCallback,
    SnapshotCallback,
    StopCallback,
    WorldModelTrainer,
)
from training.training_types import PretrainConfig


def fit_normalizer(
    dataset: Subset, encoder: JointEncoder | None, batch_size: int, workers: int, device: torch.device, world: int
) -> AudioVisualNormalizer:
    video_moments, audio_moments = RunningMoments(), RunningMoments()
    for batch in DataLoader(dataset, batch_size=batch_size, num_workers=workers):
        with torch.no_grad():
            if encoder is None:
                video, audio = batch["video"].to(device).float(), batch["audio"].to(device).float()
            else:
                video, audio = encoder.encode(batch["frames"].to(device), batch["waveform"].to(device))
        video_moments.add(video, (0, 1, 2))
        audio_moments.add(audio, (0, 1))
    if world > 1:
        for moments in (video_moments, audio_moments):
            count = torch.tensor(float(moments.count), dtype=torch.float64, device=device)
            for tensor in (moments.total, moments.squared, count):
                dist.all_reduce(tensor)
            moments.count = int(count.item())
    return AudioVisualNormalizer(*(tensor.cpu() for tensor in [*video_moments.result(), *audio_moments.result()]))


def training_arguments(config: PretrainConfig) -> TrainingArguments:
    trainer_config, optimizer_config, data_config = config.trainer, config.optimizer, config.data
    output = Path(trainer_config.output)
    return TrainingArguments(
        output_dir=str(output),
        run_name=output.name,
        per_device_train_batch_size=trainer_config.batch_size,
        gradient_accumulation_steps=trainer_config.gradient_accumulation,
        num_train_epochs=trainer_config.epochs,
        max_steps=trainer_config.max_steps or -1,
        max_grad_norm=optimizer_config.gradient_clip,
        bf16=trainer_config.precision == "bf16",
        use_cpu=trainer_config.device == "cpu",
        logging_steps=trainer_config.log_every,
        save_strategy="steps",
        eval_strategy="steps" if data_config.validation_windows else "no",
        eval_steps=trainer_config.eval_steps,
        per_device_eval_batch_size=trainer_config.batch_size,
        save_steps=trainer_config.checkpoint_steps,
        report_to=[] if trainer_config.wandb_mode == "disabled" else ["wandb"],
        dataloader_num_workers=data_config.workers,
        dataloader_prefetch_factor=4 if data_config.workers else None,
        dataloader_drop_last=True,
        remove_unused_columns=False,
        train_sampling_strategy="sequential",
        ddp_find_unused_parameters=False,
        seed=data_config.seed,
    )


def load_datasets(
    config: PretrainConfig, encoder_config: FrozenEncoderConfig, rank: int
) -> tuple[Dataset, dict[str, Dataset] | None]:
    data_config = config.data
    if not data_config.latent_cache:
        layout = ControlLayout()
        dataset = build_dataset(data_config, encoder_config, config.encoder.total_frames, layout)
        if rank == 0:
            print(f"Training records: {len(dataset.paths)} {dict(Counter(dataset.datasets))}", flush=True)
        if not data_config.validation_windows:
            return dataset, None
        return dataset, build_validation_datasets(data_config, encoder_config, config.encoder.total_frames, layout)
    cache = Path(data_config.latent_cache)
    dataset = CachedWindowDataset(cache / "train", data_config.windows)
    if rank == 0:
        print(f"Training on {len(dataset)} cached windows from {cache}", flush=True)
    if not data_config.validation_windows:
        return dataset, None
    return dataset, {
        directory.name: CachedWindowDataset(directory, data_config.validation_windows)
        for directory in sorted((cache / "validation").iterdir())
    }


def load_normalizer(
    output: Path,
    dataset: Dataset,
    encoder: JointEncoder | None,
    config: PretrainConfig,
    arguments: TrainingArguments,
) -> AudioVisualNormalizer:
    normalizer_path = output / "normalizer.pt"
    if normalizer_path.exists():
        state = torch.load(normalizer_path, map_location="cpu", weights_only=True)
        return AudioVisualNormalizer(
            state["video.mean"],
            state["video.standard_deviation"],
            state["audio.mean"],
            state["audio.standard_deviation"],
        )
    rank, world = arguments.process_index, arguments.world_size
    normalizer = fit_normalizer(
        Subset(dataset, range(rank, config.trainer.normalizer_windows, world)),
        encoder,
        config.trainer.batch_size,
        config.data.workers,
        arguments.device,
        world,
    )
    if rank == 0:
        torch.save(normalizer.state_dict(), normalizer_path)
    return normalizer


def warm_up(
    objective: WorldModelObjective,
    config: PretrainConfig,
    encoder_config: FrozenEncoderConfig,
    arguments: TrainingArguments,
) -> None:
    device, batch_size = arguments.device, config.trainer.batch_size
    time_steps = config.encoder.total_frames // encoder_config.video_frames_per_step
    patches = (encoder_config.image_size // 16) ** 2
    windows = (
        torch.randn(batch_size, time_steps, patches, encoder_config.video_latent_dimension, device=device),
        torch.randn(batch_size, time_steps, encoder_config.audio_latent_dimension, device=device),
        torch.zeros(batch_size, time_steps, ControlLayout().dimension, device=device),
        torch.zeros(batch_size, dtype=torch.int64, device=device),
    )
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=arguments.bf16):
        objective(*windows)["loss"].backward()
    for parameter in objective.parameters():
        if parameter.requires_grad and parameter.grad is None:
            parameter.requires_grad_(False)
        parameter.grad = None
    for rows in sorted({batch_size, min(4, batch_size)}) if config.trainer.compile else ():
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=arguments.bf16):
            objective(*(tensor[:rows] for tensor in windows))


def main(config: PretrainConfig) -> None:
    trainer_config, optimizer_config = config.trainer, config.optimizer
    set_recompile_limit(trainer_config.recompile_limit)
    os.environ.setdefault("WANDB_PROJECT", trainer_config.wandb_project)
    os.environ.setdefault("WANDB_MODE", trainer_config.wandb_mode)
    os.environ.setdefault("WANDB_DIR", trainer_config.output)
    os.environ.setdefault("WANDB_RUN_ID", Path(trainer_config.output).name)
    os.environ.setdefault("WANDB_RESUME", "allow")
    arguments = training_arguments(config)
    rank, world = arguments.process_index, arguments.world_size
    output = Path(trainer_config.output)
    output.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        (output / "pretrain_config.json").write_text(config.model_dump_json(indent=2))
    resume = get_last_checkpoint(str(output))

    encoder_config = build_encoder_config(config.encoder, bfloat16=trainer_config.precision == "bf16")
    dataset, validation = load_datasets(config, encoder_config, rank)
    encoder = None if config.data.latent_cache else JointEncoder(encoder_config, arguments.device)
    normalizer = load_normalizer(output, dataset, encoder, config, arguments)

    torch.manual_seed(config.data.seed)
    objective = build_objective(
        encoder_config, config.encoder.total_frames, config.model, config.objective, normalizer, arguments.device
    )
    objective.predictor.activation_checkpointing = trainer_config.activation_checkpointing
    objective.predictor.diffusion_head.activation_checkpointing = trainer_config.activation_checkpointing
    if trainer_config.compile:
        print(f"Compiled {compile_blocks(objective.predictor)} attention blocks", flush=True)
    warm_up(objective, config, encoder_config, arguments)

    batch_size = trainer_config.batch_size
    steps_per_epoch = (config.data.windows // world) // (batch_size * trainer_config.gradient_accumulation)
    total_steps = max(
        trainer_config.max_steps or steps_per_epoch * trainer_config.epochs, optimizer_config.warmup_steps + 1
    )
    optimizer = build_optimizer(
        objective, optimizer_config.name, optimizer_config.learning_rate, optimizer_config.weight_decay
    )
    schedule = build_schedule(
        optimizer,
        optimizer_config.schedule,
        optimizer_config.warmup_steps,
        total_steps,
        optimizer_config.final_learning_rate_fraction,
    )
    averaged = (
        AveragedModel(
            objective,
            multi_avg_fn=(
                warmup_ema_update(trainer_config.ema_decay)
                if trainer_config.ema_warmup
                else get_ema_multi_avg_fn(trainer_config.ema_decay)
            ),
            use_buffers=True,
        )
        if rank == 0
        else None
    )
    if averaged is not None and resume is not None:
        averaged.load_state_dict(torch.load(Path(resume) / "ema.pt", map_location="cpu", weights_only=False))
    callbacks = [
        EmaCallback(averaged, objective),
        SnapshotCallback(trainer_config.snapshot_steps),
        StopCallback(trainer_config.stop_at, Path(trainer_config.stop_file) if trainer_config.stop_file else None),
    ]
    if config.objective.slim_lsd.weight > 0:
        callbacks.append(SlimLsdCallback(objective))
    if validation:
        shown = validation.get("d2e") or next(iter(validation.values()))
        samples = default_collate([shown[index] for index in range(min(4, len(shown)))])
        callbacks.append(SampleCallback(objective, encoder, encoder_config, samples))
    trainer = WorldModelTrainer(
        model=objective,
        args=arguments,
        train_dataset=dataset,
        eval_dataset=validation,
        optimizers=(optimizer, schedule),
        callbacks=callbacks,
        encoder=encoder,
        normalizer=objective.normalizer,
        video_only=config.objective.mode == "video_only",
    )
    if rank == 0:
        print(
            f"Global batch {batch_size * world * trainer_config.gradient_accumulation}; "
            f"{steps_per_epoch} steps per epoch; resume {resume}",
            flush=True,
        )
    trainer.train(resume_from_checkpoint=resume)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("overrides", nargs="*")
    command = parser.parse_args()
    main(PretrainConfig.model_validate(load_run_settings(command.config, command.overrides)))
