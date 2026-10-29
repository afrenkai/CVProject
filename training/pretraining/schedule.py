import json
from pathlib import Path
import yaml
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from evaluation.types import ValidationRequest
from training.pretraining.configuration import PretrainingConfig, PretrainingSchedule


def pretraining_arguments(config: PretrainingConfig) -> TrainingArguments:
    settings = config.trainer
    return TrainingArguments(
        output_dir=str(config.output),
        num_train_epochs=1.0,
        max_steps=settings.maximum_steps,
        per_device_train_batch_size=settings.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=settings.accumulation_steps,
        learning_rate=settings.learning_rate,
        lr_scheduler_type="constant",
        weight_decay=0.0,
        max_grad_norm=1.0,
        eval_strategy="no",
        save_strategy="steps",
        save_steps=config.schedule.checkpoint_steps,
        save_total_limit=None,
        save_only_model=False,
        load_best_model_at_end=False,
        metric_for_best_model="unweighted_flow_loss",
        greater_is_better=False,
        label_names=["example_seed"],
        report_to="none",
        logging_steps=settings.logging_steps,
        logging_nan_inf_filter=False,
        fp16=False,
        bf16=True,
        tf32=False,
        optim="adamw_torch",
        seed=settings.seed,
        data_seed=settings.seed,
        dataloader_num_workers=settings.workers,
        remove_unused_columns=False,
        restore_callback_states_from_checkpoint=True,
        ddp_find_unused_parameters=False,
        disable_tqdm=True,
    )


def native_checkpoint_step(checkpoint: Path, world_size: int) -> int:
    names = ["model.safetensors", "optimizer.pt", "scheduler.pt", "trainer_state.json"]
    names += ["rng_state.pth"] if world_size == 1 else [f"rng_state_{rank}.pth" for rank in range(world_size)]
    assert all((checkpoint / name).is_file() for name in names), checkpoint
    step = int(json.loads((checkpoint / "trainer_state.json").read_text())["global_step"])
    assert step > 0 and checkpoint.name == f"checkpoint-{step}"
    return step


class ValidationRequests(TrainerCallback):
    def __init__(
        self, schedule: PretrainingSchedule, config_path: Path, request_directory: Path, validation_windows: int
    ) -> None:
        self.schedule = schedule
        self.config_path = config_path
        self.request_directory = request_directory
        self.validation_windows = validation_windows

    def on_save(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: object
    ) -> TrainerControl:
        if not state.is_world_process_zero or state.global_step % self.schedule.validation_steps:
            return control
        request = ValidationRequest(
            global_step=state.global_step,
            checkpoint=Path(args.output_dir) / f"checkpoint-{state.global_step}",
            config=self.config_path,
            expected_validation_windows=self.validation_windows,
            n_futures=self.schedule.n_futures,
            training_world_size=args.world_size,
            validation_world_size=self.schedule.validation_world_size,
            shard_windows=self.schedule.validation_shard_windows,
        )
        self.request_directory.mkdir(parents=True, exist_ok=True)
        destination = self.request_directory / f"step_{state.global_step}.yaml"
        pending = destination.with_suffix(".pending")
        pending.write_text(yaml.safe_dump(request.model_dump(mode="json"), sort_keys=False))
        pending.replace(destination)
        return control
