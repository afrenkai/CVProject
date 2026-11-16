import os
import time
from pathlib import Path
import psutil
import torch
from torch import Tensor, nn
from torch.optim.swa_utils import AveragedModel
from transformers import Trainer, TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint
from configs.config_types import FrozenModel
from data.data_types import DatasetConfig
from models.normalization import AudioVisualNormalizer
from training.objective import WorldModelObjective
from training.training_types import OptimizerConfig, TrainerConfig


def training_arguments(
    trainer_config: TrainerConfig, data_config: DatasetConfig, optimizer_config: OptimizerConfig
) -> TrainingArguments:
    assert trainer_config.device == "cpu" or torch.cuda.is_available(), "device: cuda but no GPU visible"
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
        eval_strategy="steps" if data_config.validation_clips else "no",
        eval_steps=trainer_config.eval_steps,
        per_device_eval_batch_size=trainer_config.batch_size,
        save_steps=trainer_config.checkpoint_steps,
        save_total_limit=trainer_config.checkpoint_limit,
        report_to=[] if trainer_config.wandb_mode == "disabled" else ["wandb"],
        dataloader_num_workers=data_config.workers,
        dataloader_prefetch_factor=4 if data_config.workers else None,
        dataloader_multiprocessing_context="forkserver" if data_config.workers else None,
        dataloader_drop_last=True,
        remove_unused_columns=False,
        train_sampling_strategy="sequential",
        ddp_find_unused_parameters=False,
        seed=data_config.seed,
    )


def start_run(config: FrozenModel, trainer_config: TrainerConfig, arguments: TrainingArguments) -> str | None:
    os.environ.setdefault("WANDB_PROJECT", trainer_config.wandb_project)
    os.environ.setdefault("WANDB_MODE", trainer_config.wandb_mode)
    os.environ.setdefault("WANDB_DIR", trainer_config.output)
    os.environ.setdefault("WANDB_RUN_ID", Path(trainer_config.output).name)
    os.environ.setdefault("WANDB_RESUME", "allow")
    output = Path(trainer_config.output)
    output.mkdir(parents=True, exist_ok=True)
    if arguments.process_index == 0:
        (output / "config.json").write_text(config.model_dump_json(indent=2))
    return get_last_checkpoint(str(output))


class RunningLossTrainer(Trainer):
    def __init__(self, *arguments: object, **settings: object) -> None:
        super().__init__(*arguments, **settings)
        self.running: dict[str, list[float]] = {}
        self.evaluation_running: dict[str, list[float]] = {}
        self.last_log = (time.perf_counter(), 0)

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, Tensor],
        return_outputs: bool = False,
        num_items_in_batch: Tensor | int | None = None,
    ) -> Tensor:
        outputs = model(**inputs)
        self.record(outputs, model.training)
        return outputs["loss"]

    def record(self, outputs: dict[str, Tensor], training: bool) -> None:
        running = self.running if training else self.evaluation_running
        for name, value in outputs.items():
            if name.endswith("_loss"):
                running.setdefault(name, []).append(value.detach().mean().item())

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        if "loss" in logs:
            logs.update({name: sum(values) / len(values) for name, values in self.running.items() if values})
            self.running = {}
            now, step = time.perf_counter(), self.state.global_step
            clips = (step - self.last_log[1]) * self.args.train_batch_size * self.args.gradient_accumulation_steps
            logs["clips_per_second"] = clips * self.args.world_size / max(now - self.last_log[0], 1e-6)
            self.last_log = (now, step)
            if torch.cuda.is_available():
                logs["peak_memory_gb"] = torch.cuda.max_memory_allocated() / 1e9
            process = psutil.Process()
            workers = process.children(recursive=True)
            logs["host_memory_gb"] = process.memory_info().rss / 1e9
            logs["worker_memory_gb"] = sum(worker.memory_info().rss for worker in workers) / 1e9
        prefix = next(
            (name.removesuffix("_loss") for name in logs if name.startswith("eval") and name.endswith("_loss")), None
        )
        if prefix is not None:
            logs.update(
                {
                    f"{prefix}_{name}": sum(values) / len(values)
                    for name, values in self.evaluation_running.items()
                    if values
                }
            )
            self.evaluation_running = {}
        super().log(logs, start_time)


class WorldModelTrainer(RunningLossTrainer):
    def __init__(
        self,
        *arguments: object,
        normalizer: AudioVisualNormalizer,
        **settings: object,
    ) -> None:
        super().__init__(*arguments, **settings)
        self.normalizer = normalizer
        self.evaluation_batches = 0

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, Tensor],
        return_outputs: bool = False,
        num_items_in_batch: Tensor | int | None = None,
    ) -> Tensor:
        with torch.no_grad(), torch.autocast(self.args.device.type, dtype=torch.bfloat16, enabled=self.args.bf16):
            video = self.normalizer.normalize_video(inputs["video"].float())
            audio = self.normalizer.normalize_audio(inputs["audio"].float())
        outputs = model(video, audio, inputs["actions"], inputs["action_present"])
        self.record(outputs, model.training)
        return outputs["loss"]

    def evaluate(self, *arguments: object, **settings: object) -> dict[str, float]:
        self.evaluation_batches = 0
        return super().evaluate(*arguments, **settings)

    def prediction_step(
        self,
        model: nn.Module,
        inputs: dict[str, Tensor],
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
    ) -> tuple[Tensor, None, None]:
        inputs = {name: tensor.to(self.args.device) for name, tensor in inputs.items()}
        devices = [self.args.device.index or 0] if self.args.device.type == "cuda" else []
        with torch.no_grad(), torch.random.fork_rng(devices=devices):
            torch.manual_seed(self.evaluation_batches)
            self.evaluation_batches += 1
            loss = self.compute_loss(model, inputs)
        return loss.detach(), None, None


class EmaCallback(TrainerCallback):
    def __init__(self, averaged: AveragedModel | None, objective: nn.Module) -> None:
        self.averaged = averaged
        self.objective = objective

    def on_step_end(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **settings: object
    ) -> None:
        if self.averaged is not None:
            self.averaged.update_parameters(self.objective)

    def on_save(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **settings: object
    ) -> None:
        if self.averaged is not None:
            torch.save(self.averaged.state_dict(), Path(args.output_dir) / f"checkpoint-{state.global_step}" / "ema.pt")


class SlimLsdCallback(TrainerCallback):
    def __init__(self, objective: WorldModelObjective) -> None:
        self.objective = objective

    def on_step_begin(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **settings: object
    ) -> None:
        config = self.objective.slim_lsd
        progress = min(1.0, max(0.0, (state.global_step - config.start + 1) / max(1, config.ramp_steps)))
        self.objective.slim_lsd_weight = config.weight if progress > 0 else 0.0
        self.objective.diagonal_fraction = 1.0 - (1.0 - config.diagonal_fraction) * progress
