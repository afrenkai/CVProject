import argparse
from pathlib import Path
import torch
from einops import rearrange, repeat
from torch import Tensor
from torch.utils.data import DataLoader, Subset
from data.latent_cache import CachedClipDataset
from evals.samples import ensemble_errors, kept_steps, resolve_checkpoint, rollout, save_summary
from training.objective import WorldModelObjective, checkpoint_weights, restore_objective
from training.training_types import PretrainConfig


def action_variants(objective: WorldModelObjective, batch: dict[str, Tensor]) -> dict[str, dict[str, Tensor]]:
    """
    in minibatch of actions
    we want to make sure that several ablations can be done.

    roll flips tensor along the 1st dim.

    Example:
        (assuming )
    reg_batch_arange = tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
    reg_batch_rolled = tensor([[8,1], [2,3], [4,5], [6,7]])

    """
    shuffled = torch.roll(torch.arange(batch["actions"].shape[0], device=batch["actions"].device), 1)
    return {
        "true": batch,
        "shuffled": {**batch, "actions": batch["actions"][shuffled]},
        "null": {**batch, "action_present": torch.zeros_like(batch["action_present"])},
        "audio_shuffled": {**batch, "audio": batch["audio"][shuffled]},
        "video_shuffled": {**batch, "video": batch["video"][shuffled]},
        "audio_muted": {**batch, "audio": objective.normalizer.audio.mean.expand_as(batch["audio"])},
        "video_off": {**batch, "video": objective.normalizer.video.mean.expand_as(batch["video"])},
    }


@torch.no_grad()
def sensitivity(
    objective: WorldModelObjective, batch: dict[str, Tensor], members: int, integration_steps: int, seed: int
) -> dict[str, Tensor]:
    totals = {}
    predictions = {}
    for name, variant in action_variants(objective, batch).items():
        errors = ensemble_errors(objective, variant, members, integration_steps, seed)
        totals.update({f"{name}/{key}": value for key, value in errors.items() if key.endswith(("_crps", "_copy_mae"))})
        predictions[name] = rollout(objective, variant, integration_steps, seed)
    truth = predictions["true"]
    for name in ("shuffled", "null", "audio_shuffled", "video_shuffled", "audio_muted", "video_off"):
        for modality, axes in (("video", (0, 2, 3)), ("audio", (0, 2))):
            change = (predictions[name][f"predicted_{modality}"] - truth[f"predicted_{modality}"]).square()
            error = (truth[f"predicted_{modality}"] - truth[f"true_{modality}"]).square()
            totals[f"{name}/{modality}_effect"] = change.mean(dim=axes) / error.mean(dim=axes)
    return totals


@torch.no_grad()
def retrieval(
    objective: WorldModelObjective,
    batch: dict[str, Tensor],
    candidates: int,
    members: int,
    integration_steps: int,
    seed: int,
) -> dict[str, Tensor]:
    clips = batch["video"].shape[0]
    expanded = {
        name: repeat(batch[name], "clip ... -> (member clip) ...", member=members) for name in ("video", "audio")
    }
    final_costs = []
    trajectory_costs = []
    for shift in range(candidates):
        order = torch.roll(torch.arange(clips, device=batch["video"].device), shift)
        candidate = {
            **expanded,
            "actions": repeat(batch["actions"][order], "clip ... -> (member clip) ...", member=members),
            "action_present": torch.ones(members * clips, dtype=torch.long, device=batch["video"].device),
        }
        result = rollout(objective, candidate, integration_steps, seed)
        predicted = rearrange(result["predicted_video"].float(), "(member clip) ... -> member clip ...", member=members)
        goal = result["true_video"][:clips].float()
        final_costs.append((predicted[:, :, -1] - goal[:, -1]).square().mean(dim=(0, 2, 3)))
        trajectory_costs.append((predicted - goal).square().mean(dim=(0, 2, 3, 4)))
    costs = {"final": torch.stack(final_costs, dim=1), "trajectory": torch.stack(trajectory_costs, dim=1)}
    metrics = {}
    for name, cost in costs.items():
        rank = (cost < cost[:, :1]).sum(dim=1).float() + ((cost == cost[:, :1]).sum(dim=1).float() - 1.0) / 2
        metrics[f"retrieval/{name}_top1"] = (rank < 0.5).float()
        metrics[f"retrieval/{name}_reciprocal_rank"] = 1.0 / (rank + 1.0)
    return metrics


@torch.no_grad()
def main(
    run: Path,
    checkpoint: Path,
    cache: Path,
    clips: int,
    batch_size: int,
    members: int,
    candidates: int,
    integration_steps: int,
) -> None:
    config = PretrainConfig.model_validate_json((run / "config.json").read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bfloat16 = config.trainer.precision == "bf16"
    encoder_config = config.encoder.resolve(bfloat16=bfloat16)
    objective = restore_objective(config, encoder_config, checkpoint_weights(checkpoint, True), device)
    dataset = CachedClipDataset(cache)
    present = [index for index in range(len(dataset)) if int(dataset[index]["action_present"]) == 1][:clips]
    assert len(present) >= batch_size, f"only {len(present)} clips with actions in {cache}"
    assert batch_size >= candidates, f"retrieval needs {candidates} distinct clips per batch, got {batch_size}"
    collected: dict[str, list[Tensor]] = {}
    for index, batch in enumerate(DataLoader(Subset(dataset, present), batch_size=batch_size, drop_last=True)):
        steps = config.encoder.total_frames // encoder_config.video_frames_per_step
        batch = {
            name: (
                tensor.to(device)[:, -kept_steps(name, steps, objective.audio_history_steps) :]
                if tensor.dim() > 1
                else tensor.to(device)
            )
            for name, tensor in batch.items()
        }
        batch = {**batch, "video": batch["video"].float(), "audio": batch["audio"].float()}
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bfloat16):
            metrics = sensitivity(objective, batch, members, integration_steps, seed=index)
            metrics.update(retrieval(objective, batch, candidates, members, integration_steps, seed=index))
        for name, value in metrics.items():
            collected.setdefault(name, []).append(value.float().cpu())

    summary = {}
    for name, values in sorted(collected.items()):
        stacked = torch.cat(values) if name.startswith("retrieval/") else torch.stack(values).mean(dim=0)
        summary[name] = stacked.mean().item() if name.startswith("retrieval/") else stacked.tolist()

    for modality in ("video", "audio"):
        true_crps = torch.tensor(summary[f"true/{modality}_crps"])
        for name in ("shuffled", "null", "audio_shuffled", "video_shuffled", "audio_muted", "video_off"):
            summary[f"{name}/{modality}_crps_gain"] = (
                torch.tensor(summary[f"{name}/{modality}_crps"]) / true_crps - 1.0
            ).tolist()

    # TODO: handle logging smarter
    summary["retrieval/chance_top1"] = 1.0 / candidates
    for name, value in summary.items():
        shown = value if isinstance(value, float) else [round(entry, 4) for entry in value[:: max(1, len(value) // 7)]]
        print(f"{name}: {shown}", flush=True)
    save_summary(summary, checkpoint, "action_eval", cache)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("checkpoint")
    parser.add_argument("cache", type=Path)
    parser.add_argument("--clips", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--members", type=int, default=4)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--integration-steps", type=int, default=8)
    args = parser.parse_args()
    main(
        args.run,
        resolve_checkpoint(args.run, args.checkpoint),
        args.cache,
        args.clips,
        args.batch_size,
        args.members,
        args.candidates,
        args.integration_steps,
    )
