import torch
from einops import rearrange, repeat
from torch import Tensor
from models.joint_encoder import JointEncoder
from training.objective import WorldModelObjective
from transport.solvers import solve_flow_map


def batch_latents(encoder: JointEncoder | None, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
    if "video" in batch:
        return batch["video"], batch["audio"]
    return encoder.encode(batch["frames"], batch["waveform"])


@torch.no_grad()
def rollout(
    objective: WorldModelObjective,
    encoder: JointEncoder | None,
    batch: dict[str, Tensor],
    integration_steps: int,
    seed: int,
) -> dict[str, Tensor]:
    video, audio = batch_latents(encoder, batch)
    video = objective.normalizer.normalize_video(video.float())
    audio = objective.normalizer.normalize_audio(audio.float())
    context_steps = objective.context_steps
    conditioned = objective.encode_actions(batch["actions"], batch["action_present"])
    context_video, context_audio = video[:, :context_steps], audio[:, :context_steps]
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
        True,
        conditioned[:, :context_steps],
    )
    return {
        "context_video": context_video,
        "context_audio": context_audio,
        "true_video": video[:, context_steps:],
        "true_audio": audio[:, context_steps:],
        "predicted_video": objective.transport.video_frames(sampled_video, context_video),
        "predicted_audio": sampled_audio,
    }


def horizon_errors(result: dict[str, Tensor]) -> dict[str, Tensor]:
    video_persistence = result["context_video"][:, -1:].expand_as(result["true_video"])
    audio_persistence = result["context_audio"][:, -1:].expand_as(result["true_audio"])
    return {
        "video": (result["predicted_video"] - result["true_video"]).square().mean(dim=(0, 2, 3)),
        "video_persistence": (video_persistence - result["true_video"]).square().mean(dim=(0, 2, 3)),
        "audio": (result["predicted_audio"] - result["true_audio"]).square().mean(dim=(0, 2)),
        "audio_persistence": (audio_persistence - result["true_audio"]).square().mean(dim=(0, 2)),
        "video_spread": result["predicted_video"].std(dim=(0, 2, 3)) / result["true_video"].std(dim=(0, 2, 3)),
        "audio_spread": result["predicted_audio"].std(dim=(0, 2)) / result["true_audio"].std(dim=(0, 2)),
    }


def sweep_rollouts(
    objective: WorldModelObjective,
    encoder: JointEncoder | None,
    batch: dict[str, Tensor],
    integration_steps: tuple[int, ...],
    seed: int,
) -> dict[int, dict[str, Tensor]]:
    return {
        steps: {name: tensor.float() for name, tensor in rollout(objective, encoder, batch, steps, seed).items()}
        for steps in integration_steps
    }


def sweep_metrics(errors: dict[int, dict[str, Tensor]]) -> dict[str, float]:
    reference = errors[max(errors)]
    metrics = {
        "samples/video_persistence_mse": reference["video_persistence"].mean().item(),
        "samples/audio_persistence_mse": reference["audio_persistence"].mean().item(),
    }
    for steps, values in errors.items():
        metrics[f"samples/video_mse_nfe{steps}"] = values["video"].mean().item()
        metrics[f"samples/audio_mse_nfe{steps}"] = values["audio"].mean().item()
        metrics[f"samples/video_spread_nfe{steps}"] = values["video_spread"].mean().item()
        metrics[f"samples/audio_spread_nfe{steps}"] = values["audio_spread"].mean().item()
    return metrics


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
    encoder: JointEncoder | None,
    batch: dict[str, Tensor],
    members: int,
    integration_steps: int,
    seed: int,
) -> dict[str, Tensor]:
    video, audio = batch_latents(encoder, batch)
    batch = {"video": video, "audio": audio, "actions": batch["actions"], "action_present": batch["action_present"]}
    windows = batch["video"].shape[0]
    repeated = {
        name: repeat(tensor, "window ... -> (member window) ...", member=members) for name, tensor in batch.items()
    }
    result = {
        name: tensor.float() for name, tensor in rollout(objective, encoder, repeated, integration_steps, seed).items()
    }
    totals: dict[str, Tensor] = {}
    for modality, reduce in (("video", (0, 2, 3)), ("audio", (0, 2))):
        predicted = rearrange(
            result[f"predicted_{modality}"], "(member window) ... -> member window ...", member=members
        )
        target = result[f"true_{modality}"][:windows]
        persistence = result[f"context_{modality}"][:windows, -1:].expand_as(target)
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
