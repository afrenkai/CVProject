import json
from pathlib import Path
import torch
from einops import rearrange, repeat
from torch import Tensor
from transformers.trainer_utils import get_last_checkpoint
from training.objective import WorldModelObjective
from transport.solvers import solve_flow_map


@torch.no_grad()
def rollout(
    objective: WorldModelObjective,
    batch: dict[str, Tensor],
    integration_steps: int,
    seed: int,
) -> dict[str, Tensor]:
    video = objective.normalizer.normalize_video(batch["video"].float())
    audio = objective.prepare_audio(objective.normalizer.normalize_audio(batch["audio"].float()))
    context_steps = objective.context_steps
    conditioned = objective.encode_actions(batch["actions"], batch["action_present"])
    context_video, context_audio = objective.silence(video[:, :context_steps], audio[:, :context_steps])
    generator = torch.Generator(device=video.device).manual_seed(seed)
    noise_video = torch.randn(video[:, context_steps:].shape, generator=generator, device=video.device)
    noise_audio = torch.randn(audio[:, context_steps:].shape, generator=generator, device=video.device)
    sampled_video, sampled_audio = solve_flow_map(
        objective.predictor,
        noise_video,
        noise_audio,
        context_video,
        context_audio,
        conditioned[:, context_steps:],
        objective.transport.time_grid(integration_steps, noise_video),
        conditioned[:, :context_steps],
    )
    return {
        "context_video": context_video,
        "context_audio": objective.scored_audio(audio[:, :context_steps]),
        "true_video": video[:, context_steps:],
        "true_audio": objective.scored_audio(audio[:, context_steps:]),
        "predicted_video": objective.transport.video_frames(sampled_video, context_video),
        "predicted_audio": objective.scored_audio(sampled_audio),
    }


def resolve_checkpoint(run: Path, checkpoint: str) -> Path:
    if checkpoint != "latest":
        return Path(checkpoint)
    latest = get_last_checkpoint(str(run))
    assert latest is not None, f"no checkpoint in {run}"
    return Path(latest)


def save_summary(summary: dict[str, object], checkpoint: Path, evaluation: str, cache: Path) -> None:
    path = checkpoint / f"{evaluation}_{cache.parent.parent.name}_{cache.name}.json"
    path.write_text(json.dumps(summary, indent=1))
    print(f"saved {path}", flush=True)


def ensemble_crps(samples: Tensor, target: Tensor) -> Tensor:
    members = samples.shape[0]
    skill = (samples - target).abs().mean(dim=0)
    ordered = samples.sort(dim=0).values
    ranks = torch.arange(1, members + 1, device=samples.device, dtype=samples.dtype)
    weights = (2 * ranks - members - 1).view(members, *([1] * (samples.dim() - 1)))
    dispersion = (weights * ordered).sum(dim=0) / (members * members)
    return skill - dispersion


@torch.no_grad()
def ensemble_errors(
    objective: WorldModelObjective,
    batch: dict[str, Tensor],
    members: int,
    integration_steps: int,
    seed: int,
) -> dict[str, Tensor]:
    batch = {name: batch[name] for name in ("video", "audio", "actions", "action_present")}
    clips = batch["video"].shape[0]
    repeated = {name: repeat(tensor, "clip ... -> (member clip) ...", member=members) for name, tensor in batch.items()}
    result = {name: tensor.float() for name, tensor in rollout(objective, repeated, integration_steps, seed).items()}
    totals: dict[str, Tensor] = {}
    for modality, reduce in (("video", (0, 2, 3)), ("audio", (0, 2))):
        predicted = rearrange(result[f"predicted_{modality}"], "(member clip) ... -> member clip ...", member=members)
        target = result[f"true_{modality}"][:clips]
        persistence = result[f"context_{modality}"][:clips, -1:].expand_as(target)
        member_errors = (predicted - target).square().mean(dim=tuple(axis + 1 for axis in reduce[1:]))
        ensemble_mean = predicted.mean(dim=0)
        spread = predicted.var(dim=0, unbiased=True).mean(dim=reduce).sqrt()
        skill = (ensemble_mean - target).square().mean(dim=reduce).sqrt()
        totals[f"{modality}_crps"] = ensemble_crps(predicted, target).mean(dim=reduce)
        totals[f"{modality}_copy_mae"] = (persistence - target).abs().mean(dim=reduce)
        totals[f"{modality}_copy_mse"] = (persistence - target).square().mean(dim=reduce)
        totals[f"{modality}_single_mse"] = member_errors.mean(dim=(0, 1))
        totals[f"{modality}_best_of_n_mse"] = member_errors.min(dim=0).values.mean(dim=0)
        totals[f"{modality}_ensemble_mean_mse"] = skill.square()
        totals[f"{modality}_spread"] = spread * ((members + 1) / members) ** 0.5 / skill
    return totals


def ensemble_metrics(errors: dict[str, Tensor], prefix: str) -> dict[str, float]:
    metrics = {}
    for modality in ("video", "audio"):
        copy_mse = errors[f"{modality}_copy_mse"].mean()
        metrics[f"{prefix}/{modality}_crps_vs_copy"] = (
            errors[f"{modality}_crps"].mean() / errors[f"{modality}_copy_mae"].mean()
        ).item()
        for name in ("single_mse", "best_of_n_mse", "ensemble_mean_mse"):
            metrics[f"{prefix}/{modality}_{name}_vs_copy"] = (errors[f"{modality}_{name}"].mean() / copy_mse).item()
        spread = errors[f"{modality}_spread"]
        third = max(spread.shape[0] // 3, 1)
        metrics[f"{prefix}/{modality}_spread_skill_early"] = spread[:third].mean().item()
        metrics[f"{prefix}/{modality}_spread_skill_late"] = spread[-third:].mean().item()
    return metrics
