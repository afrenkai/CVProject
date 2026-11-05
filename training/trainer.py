import shutil
import time
from pathlib import Path
import torch
import wandb
from matplotlib import pyplot
from torch import Tensor, nn
from torch.optim.swa_utils import AveragedModel
from transformers import Trainer, TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from models.encoder_types import FrozenEncoderConfig
from models.joint_encoder import JointEncoder
from models.normalization import AudioVisualNormalizer
from plotting.samples import sample_report
from training.objective import WorldModelObjective
from training.samples import batch_latents, sweep_rollouts


class WorldModelTrainer(Trainer):
    def __init__(
        self,
        *arguments: object,
        encoder: JointEncoder | None,
        normalizer: AudioVisualNormalizer,
        video_only: bool,
        **settings: object,
    ) -> None:
        super().__init__(*arguments, **settings)
        self.encoder = encoder
        self.normalizer = normalizer
        self.video_only = video_only  # to make sure audio contributed
        self.running: dict[str, list[float]] = {}
        self.evaluation_running: dict[str, list[float]] = {}
        self.evaluation_batches = 0
        self.last_log = (time.perf_counter(), 0)

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, Tensor],
        return_outputs: bool = False,
        num_items_in_batch: Tensor | int | None = None,
    ) -> Tensor:
        with torch.no_grad(), torch.autocast(self.args.device.type, dtype=torch.bfloat16, enabled=self.args.bf16):
            video, audio = batch_latents(self.encoder, inputs)
            video = self.normalizer.normalize_video(video.float())
            audio = self.normalizer.normalize_audio(audio.float())
            if self.video_only:
                audio = torch.zeros_like(audio)
        outputs = model(video, audio, inputs["actions"], inputs["action_present"])
        running = self.running if model.training else self.evaluation_running
        for name in ("video_flow_loss", "audio_flow_loss", "slim_lsd_loss"):
            if name in outputs:
                running.setdefault(name, []).append(outputs[name].mean().item())
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

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        if "loss" in logs:
            logs.update({name: sum(values) / len(values) for name, values in self.running.items() if values})
            self.running = {}
            now, step = time.perf_counter(), self.state.global_step
            windows = (step - self.last_log[1]) * self.args.train_batch_size * self.args.gradient_accumulation_steps
            logs["windows_per_second"] = windows * self.args.world_size / max(now - self.last_log[0], 1e-6)
            self.last_log = (now, step)
            if torch.cuda.is_available():
                logs["peak_memory_gb"] = torch.cuda.max_memory_allocated() / 1e9
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


class SnapshotCallback(TrainerCallback):
    def __init__(self, snapshot_steps: int) -> None:
        self.snapshot_steps = snapshot_steps

    def on_save(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **settings: object
    ) -> None:
        if not state.is_world_process_zero:
            return
        for checkpoint in Path(args.output_dir).glob("checkpoint-*"):
            step = int(checkpoint.name.removeprefix("checkpoint-"))
            if step < state.global_step and not (self.snapshot_steps and step % self.snapshot_steps == 0):
                shutil.rmtree(checkpoint)


# TODO: handle nicer. this is really ugly but slurm needs to exit gracefully since doing multi-node
class StopCallback(TrainerCallback):
    def __init__(self, stop_at: float, stop_file: Path | None) -> None:
        self.stop_at = stop_at
        self.stop_file = stop_file

    def on_step_end(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **settings: object
    ) -> None:
        deadline = bool(self.stop_at) and time.time() > self.stop_at
        requested = self.stop_file is not None and self.stop_file.exists()
        if deadline or requested:
            control.should_save = True
            control.should_training_stop = True

#TODO: ablation for later, not even used rn
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


class SampleCallback(TrainerCallback):
    def __init__(
        self,
        objective: WorldModelObjective,
        encoder: JointEncoder | None,
        encoder_config: FrozenEncoderConfig,
        batch: dict[str, Tensor],
        integration_steps: tuple[int, ...] = (1, 4, 8, 16),
    ) -> None:
        self.objective = objective
        self.encoder = encoder
        self.encoder_config = encoder_config
        self.batch = batch
        self.integration_steps = integration_steps
        self.rendered_step = -1

    def on_evaluate(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **settings: object
    ) -> None:
        if not state.is_world_process_zero or wandb.run is None or self.rendered_step == state.global_step:
            return
        self.rendered_step = state.global_step
        batch = {name: tensor.to(args.device) for name, tensor in self.batch.items()}
        training = self.objective.training
        self.objective.eval()
        with torch.autocast(args.device.type, dtype=torch.bfloat16, enabled=args.bf16):
            results = sweep_rollouts(self.objective, self.encoder, batch, self.integration_steps, seed=0)
        self.objective.train(training)
        metrics, figures = sample_report(results, batch.get("frames"), batch.get("waveform"), self.encoder_config)
        images = {f"samples/{name}": wandb.Image(figure) for name, figure in figures.items()}
        wandb.log(metrics | images | {"train/global_step": state.global_step})
        for figure in figures.values():
            pyplot.close(figure)
