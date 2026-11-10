import argparse
from pathlib import Path
import torch
from einops import rearrange, repeat
from torch import Tensor
from torch.utils.data import DataLoader
from data.latent_cache import CachedClipDataset
from evals.samples import ensemble_crps, resolve_checkpoint, rollout, save_summary
from models.encoder_types import FrozenEncoderConfig
from training.objective import WorldModelObjective, checkpoint_weights, random_objective, restore_objective
from training.training_types import PretrainConfig


def step_change(sequence: Tensor, previous: Tensor) -> Tensor:
    flat = rearrange(torch.cat((previous, sequence), dim=1), "clip time ... -> clip time (...)")
    return (flat[:, 1:] - flat[:, :-1]).norm(dim=-1)


def standardize(values: Tensor) -> Tensor:
    return (values - values.mean(dim=1, keepdim=True)) / values.std(dim=1, keepdim=True).clamp_min(1e-6)


def alignment(reference: Tensor, signal: Tensor, maximum_lag: int) -> tuple[Tensor, Tensor]:
    reference, signal = standardize(reference), standardize(signal)
    steps = reference.shape[1]
    lags = range(-maximum_lag, maximum_lag + 1)
    correlations = torch.stack(
        [
            (reference[:, max(0, -lag) : steps - max(0, lag)] * signal[:, max(0, lag) : steps - max(0, -lag)]).mean(
                dim=1
            )
            for lag in lags
        ],
        dim=1,
    )
    peak, index = correlations.max(dim=1)
    return index.float() - maximum_lag, peak


def consistency(result: dict[str, Tensor], clips: int, maximum_lag: int) -> dict[str, Tensor]:
    motion = step_change(result["true_video"][:clips].float(), result["context_video"][:clips, -1:].float())
    activity = step_change(result["true_audio"][:clips].float(), result["context_audio"][:clips, -1:].float())
    predicted_motion = step_change(
        result["predicted_video"][:clips].float(), result["context_video"][:clips, -1:].float()
    )
    predicted_activity = step_change(
        result["predicted_audio"][:clips].float(), result["context_audio"][:clips, -1:].float()
    )
    true_lag, true_peak = alignment(motion, activity, maximum_lag)
    audio_lag, audio_peak = alignment(motion, predicted_activity, maximum_lag)
    video_lag, video_peak = alignment(predicted_motion, activity, maximum_lag)
    return {
        "audio_lag_error": (audio_lag - true_lag).abs(),
        "audio_corr_error": (audio_peak - true_peak).abs(),
        "video_lag_error": (video_lag - true_lag).abs(),
        "video_corr_error": (video_peak - true_peak).abs(),
    }


@torch.no_grad()
def clip_crps(
    objective: WorldModelObjective, batch: dict[str, Tensor], members: int, integration_steps: int, seed: int
) -> dict[str, Tensor]:
    clips = batch["video"].shape[0]
    repeated = {name: repeat(tensor, "clip ... -> (member clip) ...", member=members) for name, tensor in batch.items()}
    result = rollout(objective, repeated, integration_steps, seed)
    scores = {}
    for modality, axes in (("video", (1, 2, 3)), ("audio", (1, 2))):
        predicted = rearrange(
            result[f"predicted_{modality}"].float(), "(member clip) ... -> member clip ...", member=members
        )
        target = result[f"true_{modality}"][:clips].float()
        scores[modality] = ensemble_crps(predicted, target).mean(dim=axes)
        persistence = result[f"context_{modality}"][:clips, -1:].float().expand_as(target)
        scores[f"{modality}_copy"] = (persistence - target).abs().mean(dim=axes)
    return {**scores, **consistency(result, clips, maximum_lag=5)}


def twin_objective(
    twin: Path | None, checkpoint: Path, encoder_config: FrozenEncoderConfig, device: torch.device
) -> WorldModelObjective | None:
    if twin is None:
        return None
    config = PretrainConfig.model_validate_json((twin / "config.json").read_text())
    return restore_objective(config, encoder_config, checkpoint_weights(twin / checkpoint.name, True), device)


def clip_scores(objective: WorldModelObjective, batch: dict[str, Tensor], onset_steps: int) -> dict[str, Tensor]:
    context_steps = objective.context_steps
    audio = objective.normalizer.normalize_audio(batch["audio"].float())
    video = objective.normalizer.normalize_video(batch["video"].float())
    recent = audio[:, context_steps - onset_steps : context_steps]
    onset = (recent[:, 1:] - recent[:, :-1]).norm(dim=-1).max(dim=1).values
    surprise = (video[:, context_steps:] - video[:, context_steps - 1 : context_steps]).square().mean(dim=(1, 2, 3))
    return {"onset": onset, "surprise": surprise}


def stratified(benefit: Tensor, score: Tensor) -> dict[str, float]:
    order = score.argsort()
    # assuming .numel works like np impl, no clue actually
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
    video_twin: Path | None,
    audio_twin: Path | None,
    clips: int | None,
    batch_size: int,
    members: int,
    integration_steps: int,
    onset_steps: int,
) -> None:
    config = PretrainConfig.model_validate_json((run / "config.json").read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bfloat16 = config.trainer.precision == "bf16"
    encoder_config = config.encoder.resolve(bfloat16=bfloat16)
    objective = restore_objective(config, encoder_config, checkpoint_weights(checkpoint, True), device)
    twins = {
        "video_twin": twin_objective(video_twin, checkpoint, encoder_config, device),
        "audio_twin": twin_objective(audio_twin, checkpoint, encoder_config, device),
        "random_init": random_objective(
            config, encoder_config, checkpoint_weights(checkpoint, True), device, config.data.seed
        ),
    }
    steps = config.encoder.total_frames // encoder_config.video_frames_per_step
    collected: dict[str, list[Tensor]] = {}
    for index, batch in enumerate(DataLoader(CachedClipDataset(cache, clips), batch_size=batch_size, drop_last=True)):
        batch = {
            name: tensor.to(device)[:, -steps:].float() if tensor.dim() > 1 else tensor.to(device)
            for name, tensor in batch.items()
        }
        shuffled = torch.roll(torch.arange(batch_size, device=device), 1)
        variants = {
            "true": batch,
            "audio_shuffled": {**batch, "audio": batch["audio"][shuffled]},
            "audio_muted": {**batch, "audio": objective.normalizer.audio.mean.expand_as(batch["audio"])},
            "video_shuffled": {**batch, "video": batch["video"][shuffled]},
            "video_off": {**batch, "video": objective.normalizer.video.mean.expand_as(batch["video"])},
        }
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bfloat16):
            crps = {
                name: clip_crps(objective, variant, members, integration_steps, index)
                for name, variant in variants.items()
            }
            for name, twin in twins.items():
                if twin is not None:
                    crps[name] = clip_crps(twin, batch, members, integration_steps, index)
        values = {
            **{
                f"{modality}_crps_{name}": scores[modality]
                for name, scores in crps.items()
                for modality in ("video", "audio")
            },
            **{f"{modality}_crps_copy": crps["true"][f"{modality}_copy"] for modality in ("video", "audio")},
            **{
                f"{metric}_{name}": scores[metric]
                for name, scores in crps.items()
                for metric in ("audio_lag_error", "audio_corr_error", "video_lag_error", "video_corr_error")
            },
            **clip_scores(objective, batch, onset_steps),
        }
        for name, value in values.items():
            collected.setdefault(name, []).append(value.float().cpu())
    tensors = {name: torch.cat(values) for name, values in collected.items()}
    summary: dict[str, object] = {"clips": tensors["video_crps_true"].numel()}
    comparisons = {
        "video": ("audio_shuffled", "audio_muted", "video_twin", "random_init", "copy"),
        "audio": ("video_shuffled", "video_off", "audio_twin", "random_init", "copy"),
    }
    for modality, baselines in comparisons.items():
        true_crps = tensors[f"{modality}_crps_true"]
        summary[f"{modality}/crps"] = true_crps.mean().item()
        for baseline in baselines:
            if f"{modality}_crps_{baseline}" not in tensors:
                continue
            benefit = tensors[f"{modality}_crps_{baseline}"] - true_crps
            prefix = f"{modality}/{baseline}"
            summary[f"{prefix}/mean_benefit"] = benefit.mean().item()
            summary[f"{prefix}/relative_benefit"] = (benefit / true_crps).mean().item()
            summary[f"{prefix}/by_onset"] = stratified(benefit, tensors["onset"])
            summary[f"{prefix}/by_surprise"] = stratified(benefit, tensors["surprise"])
    for name in crps:
        for metric in ("audio_lag_error", "audio_corr_error", "video_lag_error", "video_corr_error"):
            summary[f"consistency/{name}/{metric}"] = tensors[f"{metric}_{name}"].mean().item()
    for name, value in summary.items():
        print(f"{name}: {value}", flush=True)
    save_summary(summary, checkpoint, "cross_modal", cache)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("checkpoint")
    parser.add_argument("cache", type=Path)
    parser.add_argument("--video-twin", type=Path)
    parser.add_argument("--audio-twin", type=Path)
    parser.add_argument("--clips", type=int)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--members", type=int, default=4)
    parser.add_argument("--integration-steps", type=int, default=8)
    parser.add_argument("--onset-steps", type=int, default=10)
    args = parser.parse_args()
    main(
        args.run,
        resolve_checkpoint(args.run, args.checkpoint),
        args.cache,
        args.video_twin,
        args.audio_twin,
        args.clips,
        args.batch_size,
        args.members,
        args.integration_steps,
        args.onset_steps,
    )
