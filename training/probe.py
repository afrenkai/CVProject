import json
from pathlib import Path
import torch
import wandb
from torch import Tensor
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from data.latent_cache import CachedClipDataset
from evals.audio_anticipation import clip_crps
from evals.samples import kept_steps
from models.normalization import AudioVisualNormalizer
from training.training_types import ProbeConfig


def probe_variants(batch: dict[str, Tensor], normalizer: AudioVisualNormalizer) -> dict[str, dict[str, Tensor]]:
    shuffled = torch.roll(torch.arange(batch["video"].shape[0], device=batch["video"].device), 1)
    return {
        "true": batch,
        "video_shuffled": {**batch, "video": batch["video"][shuffled]},
        "video_off": {**batch, "video": normalizer.video.mean.expand_as(batch["video"])},
        "audio_shuffled": {**batch, "audio": batch["audio"][shuffled]},
        "audio_muted": {**batch, "audio": normalizer.audio.mean.expand_as(batch["audio"])},
    }


def relative_gains(scores: dict[str, Tensor], modality: str, ablations: tuple[str, str]) -> dict[str, float]:
    true = scores[f"true/{modality}"]
    gains = {name: ((scores[f"{name}/{modality}"] - true) / true).mean().item() for name in ablations}
    content = gains[ablations[0]] / gains[ablations[1]] if abs(gains[ablations[1]]) > 1e-9 else 0.0
    return {
        f"{modality}_crps": true.mean().item(),
        f"{modality}_copy_gain": ((scores[f"true/{modality}_copy"] - true) / true).mean().item(),
        **{f"{modality}_{name}_gain": value for name, value in gains.items()},
        f"{modality}_content_ratio": content,
    }


class ProbeCallback(TrainerCallback):
    def __init__(self, config: ProbeConfig, averaged: AveragedModel | None, steps: int, output: Path) -> None:
        self.config = config
        self.averaged = averaged
        self.steps = steps
        self.output = output

    def measure(self, cache: Path, device: torch.device, bfloat16: bool) -> dict[str, float]:
        objective = self.averaged.module
        collected: dict[str, list[Tensor]] = {}
        loader = DataLoader(
            CachedClipDataset(cache, self.config.clips), batch_size=self.config.batch_size, drop_last=True
        )
        for index, batch in enumerate(loader):
            batch = {
                name: (
                    tensor.to(device)[:, -kept_steps(name, self.steps, objective.audio_history_steps) :].float()
                    if tensor.dim() > 1
                    else tensor.to(device)
                )
                for name, tensor in batch.items()
            }
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bfloat16):
                for name, variant in probe_variants(batch, objective.normalizer).items():
                    scores = clip_crps(objective, variant, self.config.members, self.config.integration_steps, index)
                    for metric in ("audio", "audio_copy", "video", "video_copy"):
                        collected.setdefault(f"{name}/{metric}", []).append(scores[metric].float().cpu())
        scores = {name: torch.cat(values) for name, values in collected.items()}
        return {
            **relative_gains(scores, "audio", ("video_shuffled", "video_off")),
            **relative_gains(scores, "video", ("audio_shuffled", "audio_muted")),
        }

    def on_step_end(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **settings: object
    ) -> None:
        if self.averaged is None or state.global_step % self.config.every:
            return
        objective = self.averaged.module
        objective.eval()
        metrics = {
            f"probe/{Path(cache).name}/{name}": value
            for cache in self.config.caches
            for name, value in self.measure(Path(cache), args.device, args.bf16).items()
        }
        print(json.dumps({"step": state.global_step, **metrics}), flush=True)
        with (self.output / "probe.jsonl").open("a") as handle:
            handle.write(json.dumps({"step": state.global_step, **metrics}) + "\n")
        if wandb.run is not None:
            wandb.log({**metrics, "train/global_step": state.global_step})
        if self.config.snapshot:
            snapshot = self.output / f"snapshot-{state.global_step}"
            snapshot.mkdir(exist_ok=True)
            torch.save(self.averaged.state_dict(), snapshot / "ema.pt")
