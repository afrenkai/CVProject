import argparse
import json
from pathlib import Path
import torch
from einops import rearrange, repeat
from torch import Tensor
from torch.utils.data import DataLoader
from data.latent_cache import CachedWindowDataset
from evals.samples import ensemble_crps, rollout
from models.joint_encoder import build_encoder_config
from training.objective import WorldModelObjective, checkpoint_weights, restore_objective
from training.training_types import PretrainConfig


@torch.no_grad()
def window_crps(
    objective: WorldModelObjective, batch: dict[str, Tensor], members: int, integration_steps: int, seed: int
) -> Tensor:
    windows = batch["video"].shape[0]
    repeated = {
        name: repeat(tensor, "window ... -> (member window) ...", member=members) for name, tensor in batch.items()
    }
    result = rollout(objective, repeated, integration_steps, seed)
    predicted = rearrange(result["predicted_video"].float(), "(member window) ... -> member window ...", member=members)
    target = result["true_video"][:windows].float()
    return ensemble_crps(predicted, target).mean(dim=(1, 2, 3))


def window_scores(objective: WorldModelObjective, batch: dict[str, Tensor], onset_steps: int) -> dict[str, Tensor]:
    context_steps = objective.context_steps
    audio = objective.normalizer.normalize_audio(batch["audio"].float())
    video = objective.normalizer.normalize_video(batch["video"].float())
    recent = audio[:, context_steps - onset_steps : context_steps]
    onset = (recent[:, 1:] - recent[:, :-1]).norm(dim=-1).max(dim=1).values
    surprise = (video[:, context_steps:] - video[:, context_steps - 1 : context_steps]).square().mean(dim=(1, 2, 3))
    return {"onset": onset, "surprise": surprise}


def stratified(benefit: Tensor, score: Tensor) -> dict[str, float]:
    order = score.argsort()
    #assuming .numel works like np impl, no clue actually
    quarter = max(1, order.numel() // 4)
    ranks = torch.empty_like(score)
    ranks[order] = torch.arange(score.numel(), dtype=score.dtype)
    benefit_ranks = torch.empty_like(benefit)
    benefit_ranks[benefit.argsort()] = torch.arange(benefit.numel(), dtype=benefit.dtype)
    return {
        "top_quarter": benefit[order[-quarter:]].mean().item(),
        "bottom_quarter": benefit[order[:quarter]].mean().item(),
        "spearman": torch.corrcoef(torch.stack((ranks, benefit_ranks)))[0, 1].item(),
    }


@torch.no_grad()
def main(
    run: Path,
    checkpoint: Path,
    cache: Path,
    twin: Path | None,
    batch_size: int,
    members: int,
    integration_steps: int,
    onset_steps: int,
) -> None:
    config = PretrainConfig.model_validate_json((run / "pretrain_config.json").read_text())
    device = torch.device("cuda")
    bfloat16 = config.trainer.precision == "bf16"
    encoder_config = build_encoder_config(config.encoder, bfloat16=bfloat16)
    objective = restore_objective(config, encoder_config, checkpoint_weights(checkpoint, True), device)
    twin_objective = None
    if twin is not None:
        twin_config = PretrainConfig.model_validate_json((twin / "pretrain_config.json").read_text())
        twin_objective = restore_objective(
            twin_config, encoder_config, checkpoint_weights(twin / checkpoint.name, True), device
        )
    steps = config.encoder.total_frames // encoder_config.video_frames_per_step
    collected: dict[str, list[Tensor]] = {}
    for index, batch in enumerate(DataLoader(CachedWindowDataset(cache), batch_size=batch_size, drop_last=True)):
        batch = {
            name: tensor.to(device)[:, -steps:].float() if tensor.dim() > 1 else tensor.to(device)
            for name, tensor in batch.items()
        }
        shuffled = torch.roll(torch.arange(batch_size, device=device), 1)
        variants = {
            "true": batch,
            "audio_shuffled": {**batch, "audio": batch["audio"][shuffled]},
        }
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bfloat16):
            crps = {
                name: window_crps(objective, variant, members, integration_steps, index)
                for name, variant in variants.items()
            }
            if twin_objective is not None:
                silent = {**batch, "audio": torch.zeros_like(batch["audio"])}
                crps["twin"] = window_crps(twin_objective, silent, members, integration_steps, index)
        values = {
            **{f"crps_{name}": value for name, value in crps.items()},
            **window_scores(objective, batch, onset_steps),
        }
        for name, value in values.items():
            collected.setdefault(name, []).append(value.float().cpu())
    tensors = {name: torch.cat(values) for name, values in collected.items()}
    summary: dict[str, object] = {"windows": tensors["crps_true"].numel()}
    benefits = {"shuffle": tensors["crps_audio_shuffled"] - tensors["crps_true"]}
    if "crps_twin" in tensors:
        benefits["twin"] = tensors["crps_twin"] - tensors["crps_true"]
    for name, benefit in benefits.items():
        summary[f"{name}/mean_benefit"] = benefit.mean().item()
        summary[f"{name}/relative_benefit"] = (benefit / tensors["crps_true"]).mean().item()
        summary[f"{name}/by_onset"] = stratified(benefit, tensors["onset"])
        summary[f"{name}/by_surprise"] = stratified(benefit, tensors["surprise"])
    output = checkpoint / f"audio_anticipation_{cache.parent.parent.name}.json"
    output.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"saved {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("cache", type=Path)
    parser.add_argument("--twin", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--members", type=int, default=4)
    parser.add_argument("--integration-steps", type=int, default=8)
    parser.add_argument("--onset-steps", type=int, default=10)
    command = parser.parse_args()
    main(
        command.run,
        command.checkpoint,
        command.cache,
        command.twin,
        command.batch_size,
        command.members,
        command.integration_steps,
        command.onset_steps,
    )
