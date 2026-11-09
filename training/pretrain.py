import argparse
import os
from pathlib import Path
import torch
import torch.distributed as dist
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import TrainingArguments, get_cosine_with_min_lr_schedule_with_warmup
from transformers.trainer_utils import get_last_checkpoint
from configs.loading import load_run_settings
from data.control_types import ControlLayout
from data.latent_cache import CachedWindowDataset
from models.encoder_types import FrozenEncoderConfig
from models.joint_encoder import build_encoder_config
from models.normalization import AudioVisualNormalizer, RunningMoments
from models.transformers import AdaptiveAttentionBlock
from training.objective import WorldModelObjective, build_objective, checkpoint_weights
from training.optimization import build_optimizer, warmup_ema_update
from training.trainer import (
    EmaCallback,
    SlimLsdCallback,
    StopCallback,
    WorldModelTrainer,
)
from training.training_types import PretrainConfig


def fit_normalizer(
    dataset: Subset, batch_size: int, workers: int, device: torch.device, world: int
) -> AudioVisualNormalizer:
    video_moments, audio_moments = RunningMoments(), RunningMoments()
    for batch in DataLoader(dataset, batch_size=batch_size, num_workers=workers):
        video_moments.add(batch["video"].to(device).float(), (0, 1, 2))
        audio_moments.add(batch["audio"].to(device).float(), (0, 1))
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
        save_total_limit=trainer_config.checkpoint_limit,
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
    cache = Path(data_config.latent_cache)
    dataset = CachedWindowDataset(cache / "train", data_config.windows, encoder_config)
    if rank == 0:
        print(f"Training on {len(dataset)} cached windows from {cache}", flush=True)
    if not data_config.validation_windows:
        return dataset, None
    return dataset, {
        directory.name: CachedWindowDataset(directory, data_config.validation_windows, encoder_config)
        for directory in sorted((cache / "validation").iterdir())
    }


def load_normalizer(
    output: Path,
    dataset: Dataset,
    config: PretrainConfig,
    arguments: TrainingArguments,
) -> AudioVisualNormalizer:
    normalizer_path = output / "normalizer.pt"
    if not normalizer_path.exists() and config.trainer.initialize_from:
        normalizer_path = Path(config.trainer.initialize_from).parent / "normalizer.pt"
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
    normalizer = load_normalizer(output, dataset, config, arguments)

    torch.manual_seed(config.data.seed)
    objective = build_objective(
        encoder_config, config.encoder.total_frames, config.model, config.objective, normalizer, arguments.device
    )
    if resume is None and trainer_config.initialize_from:
        objective.load_state_dict(checkpoint_weights(Path(trainer_config.initialize_from), True))
        print(f"Initialized from {trainer_config.initialize_from}", flush=True)
    objective.predictor.activation_checkpointing = trainer_config.activation_checkpointing
    objective.predictor.diffusion_head.activation_checkpointing = trainer_config.activation_checkpointing
    torch._dynamo.config.recompile_limit = trainer_config.recompile_limit
    torch._dynamo.config.accumulated_recompile_limit = max(
        trainer_config.recompile_limit, torch._dynamo.config.accumulated_recompile_limit
    )
    if trainer_config.compile:
        for block in objective.predictor.modules():
            if isinstance(block, AdaptiveAttentionBlock):
                block.compile(dynamic=False)
    warm_up(objective, config, encoder_config, arguments)

    batch_size = trainer_config.batch_size
    steps_per_epoch = (config.data.windows // world) // (batch_size * trainer_config.gradient_accumulation)
    total_steps = max(
        trainer_config.max_steps or steps_per_epoch * trainer_config.epochs, optimizer_config.warmup_steps + 1
    )
    optimizer = build_optimizer(
        objective, optimizer_config.name, optimizer_config.learning_rate, optimizer_config.weight_decay
    )
    schedule = get_cosine_with_min_lr_schedule_with_warmup(
        optimizer, optimizer_config.warmup_steps, total_steps, min_lr_rate=optimizer_config.final_learning_rate_fraction
    )
    averaged = (
        AveragedModel(
            objective,
            multi_avg_fn=warmup_ema_update(trainer_config.ema_decay),
            use_buffers=True,
        )
        if rank == 0
        else None
    )
    if averaged is not None and resume is not None:
        averaged.load_state_dict(torch.load(Path(resume) / "ema.pt", map_location="cpu", weights_only=False))
    callbacks = [
        EmaCallback(averaged, objective),
        StopCallback(trainer_config.stop_at, Path(trainer_config.stop_file) if trainer_config.stop_file else None),
    ]
    if config.objective.slim_lsd.weight > 0:
        callbacks.append(SlimLsdCallback(objective))
    trainer = WorldModelTrainer(
        model=objective,
        args=arguments,
        train_dataset=dataset,
        eval_dataset=validation,
        optimizers=(optimizer, schedule),
        callbacks=callbacks,
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
