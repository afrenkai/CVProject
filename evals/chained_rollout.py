import argparse
from pathlib import Path
import torch
from einops import rearrange, repeat
from torch import Tensor
from torch.utils.data import DataLoader
from data.latent_cache import CachedClipDataset
from evals.samples import ensemble_crps, resolve_checkpoint, save_summary
from training.objective import WorldModelObjective, checkpoint_weights, restore_objective
from training.training_types import PretrainConfig
from transport.solvers import solve_flow_map


@torch.no_grad()
def chained_rollout(
    objective: WorldModelObjective, batch: dict[str, Tensor], chunks: int, integration_steps: int, seed: int
) -> dict[str, Tensor]:
    video = objective.normalizer.normalize_video(batch["video"].float())
    audio = objective.prepare_audio(objective.normalizer.normalize_audio(batch["audio"].float()))
    context_steps, history_steps = objective.context_steps, objective.audio_history_steps
    target_steps = (video.shape[1] - context_steps) // chunks
    assert context_steps + chunks * target_steps == video.shape[1]
    assert audio.shape[1] == video.shape[1] + history_steps
    conditioned = objective.encode_actions(batch["actions"], batch["action_present"])
    generator = torch.Generator(device=video.device).manual_seed(seed)
    video_sequence, audio_sequence = video[:, :context_steps], audio[:, : history_steps + context_steps]
    for chunk in range(chunks):
        context_video, context_audio = objective.silence(
            video_sequence[:, -context_steps:], audio_sequence[:, -(history_steps + context_steps) :]
        )
        window = conditioned[:, chunk * target_steps : chunk * target_steps + context_steps + target_steps]
        noise_video = torch.randn(
            (video.shape[0], target_steps, *video.shape[2:]), generator=generator, device=video.device
        )
        noise_audio = torch.randn(
            (audio.shape[0], target_steps, audio.shape[-1]), generator=generator, device=video.device
        )
        sampled_video, sampled_audio = solve_flow_map(
            objective.predictor,
            noise_video,
            noise_audio,
            context_video,
            context_audio,
            window[:, context_steps:],
            objective.transport.time_grid(integration_steps, noise_video),
            window[:, :context_steps],
        )
        video_sequence = torch.cat(
            (video_sequence, objective.transport.video_frames(sampled_video, context_video)), dim=1
        )
        audio_sequence = torch.cat((audio_sequence, sampled_audio), dim=1)
    audio_context_steps = history_steps + context_steps
    return {
        "context_video": video[:, :context_steps],
        "context_audio": objective.scored_audio(audio[:, history_steps:audio_context_steps]),
        "true_video": video[:, context_steps:],
        "true_audio": objective.scored_audio(audio[:, audio_context_steps:]),
        "predicted_video": video_sequence[:, context_steps:],
        "predicted_audio": objective.scored_audio(audio_sequence[:, audio_context_steps:]),
    }


@torch.no_grad()
def chunk_scores(
    objective: WorldModelObjective,
    batch: dict[str, Tensor],
    chunks: int,
    members: int,
    integration_steps: int,
    seed: int,
) -> dict[str, Tensor]:
    clips = batch["video"].shape[0]
    repeated = {name: repeat(tensor, "clip ... -> (member clip) ...", member=members) for name, tensor in batch.items()}
    result = chained_rollout(objective, repeated, chunks, integration_steps, seed)
    scores = {}
    for modality, axes in (("video", (2, 3, 4)), ("audio", (2, 3))):
        predicted = rearrange(
            result[f"predicted_{modality}"].float(), "(member clip) ... -> member clip ...", member=members
        )
        target = result[f"true_{modality}"][:clips].float()
        crps = rearrange(ensemble_crps(predicted, target), "clip (chunk time) ... -> clip chunk time ...", chunk=chunks)
        persistence = result[f"context_{modality}"][:clips, -1:].float().expand_as(target)
        copy = rearrange((persistence - target).abs(), "clip (chunk time) ... -> clip chunk time ...", chunk=chunks)
        scores[modality] = crps.mean(dim=axes)
        scores[f"{modality}_copy"] = copy.mean(dim=axes)
    return scores


def ablations(objective: WorldModelObjective, batch: dict[str, Tensor]) -> dict[str, dict[str, Tensor]]:
    shuffled = torch.roll(torch.arange(batch["video"].shape[0], device=batch["video"].device), 1)
    return {
        "true": batch,
        "video_off": {**batch, "video": objective.normalizer.video.mean.expand_as(batch["video"])},
        "video_shuffled": {**batch, "video": batch["video"][shuffled]},
        "audio_muted": {**batch, "audio": objective.normalizer.audio.mean.expand_as(batch["audio"])},
        "audio_shuffled": {**batch, "audio": batch["audio"][shuffled]},
    }


def context_only(
    batch: dict[str, Tensor], original: dict[str, Tensor], objective: WorldModelObjective
) -> dict[str, Tensor]:
    context_steps = objective.context_steps
    audio_context_steps = context_steps + objective.audio_history_steps
    return {
        **batch,
        "video": torch.cat((batch["video"][:, :context_steps], original["video"][:, context_steps:]), dim=1),
        "audio": torch.cat(
            (batch["audio"][:, :audio_context_steps], original["audio"][:, audio_context_steps:]), dim=1
        ),
    }


@torch.no_grad()
def main(
    run: Path,
    checkpoint: Path,
    cache: Path,
    chunks: int,
    clips: int | None,
    batch_size: int,
    members: int,
    integration_steps: int,
) -> None:
    config = PretrainConfig.model_validate_json((run / "config.json").read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bfloat16 = config.trainer.precision == "bf16"
    encoder_config = config.encoder.resolve(bfloat16=bfloat16)
    objective = restore_objective(config, encoder_config, checkpoint_weights(checkpoint, True), device)
    collected: dict[str, list[Tensor]] = {}
    for index, batch in enumerate(DataLoader(CachedClipDataset(cache, clips), batch_size=batch_size, drop_last=True)):
        batch = {
            name: tensor.to(device).float() if tensor.dim() > 1 else tensor.to(device) for name, tensor in batch.items()
        }
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bfloat16):
            for name, variant in ablations(objective, batch).items():
                scores = chunk_scores(
                    objective, context_only(variant, batch, objective), chunks, members, integration_steps, index
                )
                for metric, value in scores.items():
                    collected.setdefault(f"{name}/{metric}", []).append(value.float().cpu())
    tensors = {name: torch.cat(values) for name, values in collected.items()}
    summary: dict[str, object] = {"clips": tensors["true/video"].shape[0], "chunks": chunks}
    for modality, baselines in (
        ("audio", ("video_off", "video_shuffled")),
        ("video", ("audio_muted", "audio_shuffled")),
    ):
        true_crps = tensors[f"true/{modality}"]
        summary[f"{modality}/crps"] = true_crps.mean(dim=0).tolist()
        summary[f"{modality}/copy_gain"] = ((tensors[f"true/{modality}_copy"] - true_crps) / true_crps).mean(0).tolist()
        for baseline in baselines:
            gain = (tensors[f"{baseline}/{modality}"] - true_crps) / true_crps
            summary[f"{modality}/{baseline}_gain"] = gain.mean(dim=0).tolist()
    for name, value in summary.items():
        print(f"{name}: {value}", flush=True)
    save_summary(summary, checkpoint, f"chained_{chunks}", cache)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("checkpoint")
    parser.add_argument("cache", type=Path)
    parser.add_argument("--chunks", type=int, default=2)
    parser.add_argument("--clips", type=int)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--members", type=int, default=4)
    parser.add_argument("--integration-steps", type=int, default=8)
    args = parser.parse_args()
    main(
        args.run,
        resolve_checkpoint(args.run, args.checkpoint),
        args.cache,
        args.chunks,
        args.clips,
        args.batch_size,
        args.members,
        args.integration_steps,
    )
