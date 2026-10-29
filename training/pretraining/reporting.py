import math
import os
from pathlib import Path
from typing import cast
import torch
import yaml
from torch import nn
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from losses.loss_types import LossReportingModel


class PretrainingMetrics(TrainerCallback):
    def __init__(self, output: Path) -> None:
        self.output = output
        self.finite_gradient_steps = 0

    def on_pre_optimizer_step(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: object
    ) -> TrainerControl:
        model = cast(nn.Module, kwargs["model"])
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        assert gradients and all(
            gradient.dtype == torch.float32 and torch.isfinite(gradient).all() for gradient in gradients
        )
        self.finite_gradient_steps += 1
        return control

    def record_interval(self, state: TrainerState, model: LossReportingModel) -> None:
        if not model.training_sample_count:
            return
        components = {
            name: (value / model.training_sample_count).item() for name, value in model.training_loss_sums.items()
        }
        assert all(math.isfinite(value) for value in components.values())
        rank = int(os.environ.get("RANK", "0"))
        report = {
            "global_step": state.global_step,
            "rank": rank,
            "sample_count": model.training_sample_count,
            "scope": "Local-rank interval mean; combine ranks using sample counts",
            "components": components,
        }
        with (self.output / f"training_components_rank_{rank}.yaml").open("a") as stream:
            yaml.safe_dump(report, stream, sort_keys=False, explicit_start=True)
        model.training_loss_sums = {}
        model.training_sample_count = 0

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, float] | None = None,
        **kwargs: object,
    ) -> TrainerControl:
        if logs is not None and "loss" in logs:
            self.record_interval(state, cast(LossReportingModel, kwargs["model"]))
        return control

    def on_train_end(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: object
    ) -> TrainerControl:
        model = cast(nn.Module, kwargs["model"])
        assert all(parameter.dtype == torch.float32 for parameter in model.parameters())
        self.record_interval(state, cast(LossReportingModel, model))
        rank = int(os.environ.get("RANK", "0"))
        report = {
            "global_step": state.global_step,
            "finite_gradient_steps": self.finite_gradient_steps,
            "parameter_precision": "float32",
            "autocast_bfloat16": args.bf16,
            "tf32": False,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
            "peak_reserved_bytes": torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0,
        }
        (self.output / f"precision_rank_{rank}.yaml").write_text(yaml.safe_dump(report, sort_keys=False))
        return control
