import os
import time
from pathlib import Path
from accelerate import Accelerator
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from training.pretraining.configuration import AllocationRuntime


def allocation_deadline(config: AllocationRuntime) -> float | None:
    if config.deadline_epoch is not None:
        return config.deadline_epoch
    end = os.environ.get("SLURM_JOB_END_TIME")
    if end is not None:
        return float(end) - config.graceful_stop_margin_seconds
    if "SLURM_JOB_ID" in os.environ:
        raise ValueError("Set SLURM_JOB_END_TIME or runtime.deadline_epoch for an allocation-boundary save")
    return None


class AllocationProgress(TrainerCallback):
    def __init__(
        self,
        accelerator: Accelerator,
        expected_start: int,
        pause_step: int,
        deadline_epoch: float | None,
        stop_file: Path | None,
    ) -> None:
        self.accelerator = accelerator
        self.expected_start = expected_start
        self.pause_step = pause_step
        self.deadline_epoch = deadline_epoch
        self.stop_file = stop_file

    def on_train_begin(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: object
    ) -> TrainerControl:
        assert state.global_step == self.expected_start
        return control

    def on_step_end(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: object
    ) -> TrainerControl:
        expired = self.deadline_epoch is not None and time.time() >= self.deadline_epoch
        requested = self.stop_file is not None and self.stop_file.exists()
        if expired or requested or state.global_step >= self.pause_step:
            self.accelerator.set_trigger()
        if self.accelerator.check_trigger():
            control.should_save = True
            control.should_training_stop = True
        return control
