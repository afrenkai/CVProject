import argparse
from pathlib import Path
import torch
from einops import rearrange, repeat
from torch import Tensor
from data.control_types import ControlLayout
from data.latent_cache import CachedClipDataset
from evals.eval_types import ActionSearchConfig, CostModality
from evals.samples import kept_steps, resolve_checkpoint, save_summary
from training.objective import WorldModelObjective, checkpoint_weights, restore_objective
from training.training_types import PretrainConfig
from transport.solvers import solve_flow_map


@torch.no_grad()
def predict(
    objective: WorldModelObjective,
    video: Tensor,
    audio: Tensor,
    controls: Tensor,
    noise_video: Tensor,
    noise_audio: Tensor,
    integration_steps: int,
) -> tuple[Tensor, Tensor]:
    candidates = controls.shape[0]
    video, audio = objective.silence(video, audio)
    context_steps = objective.context_steps
    conditioned = objective.encode_actions(controls, torch.ones(candidates, dtype=torch.long, device=controls.device))
    context_video = repeat(video[:, :context_steps], "one ... -> (one candidate) ...", candidate=candidates)
    audio_context_steps = context_steps + objective.audio_history_steps
    context_audio = repeat(audio[:, :audio_context_steps], "one ... -> (one candidate) ...", candidate=candidates)
    sampled_video, sampled_audio = solve_flow_map(
        objective.predictor,
        noise_video.expand(candidates, *noise_video.shape[1:]),
        noise_audio.expand(candidates, *noise_audio.shape[1:]),
        context_video,
        context_audio,
        conditioned[:, context_steps:],
        objective.transport.time_grid(integration_steps, noise_video),
        conditioned[:, :context_steps],
    )
    return objective.transport.video_frames(sampled_video, context_video), sampled_audio


def plan_cost(
    objective: WorldModelObjective,
    predicted: tuple[Tensor, Tensor],
    video: Tensor,
    audio: Tensor,
    context_steps: int,
    modality: CostModality,
) -> Tensor:
    video_cost = (predicted[0] - video[:, context_steps:]).square().mean(dim=(1, 2, 3))
    audio_cost = (
        (
            objective.scored_audio(predicted[1])
            - objective.scored_audio(audio[:, context_steps + objective.audio_history_steps :])
        )
        .square()
        .mean(dim=(1, 2))
    )
    match modality:
        case "video":
            return video_cost
        case "audio":
            return audio_cost
        case "audio_video":
            return video_cost + audio_cost


def candidate_controls(
    context: Tensor, keys: Tensor, axes: Tensor, buttons: Tensor, layout: ControlLayout, chunk_steps: int
) -> Tensor:
    candidates, chunks = keys.shape[:2]
    target = torch.zeros(candidates, chunks, layout.dimension, device=keys.device)
    target[:, :, buttons] = keys
    button_count = layout.keyboard_count + len(layout.mouse_buttons)
    target[:, :, button_count : button_count + 2] = torch.sinh(axes)
    steps = repeat(target, "candidate chunk control -> candidate (chunk step) control", step=chunk_steps)
    return torch.cat((repeat(context, "time control -> candidate time control", candidate=candidates), steps), dim=1)


# TODO: ugly, refactor and ideally remove
def chunk_summary(controls: Tensor, buttons: Tensor, layout: ControlLayout, chunk_steps: int) -> tuple[Tensor, Tensor]:
    chunked = rearrange(controls, "(chunk step) control -> chunk step control", step=chunk_steps)
    button_count = layout.keyboard_count + len(layout.mouse_buttons)
    return chunked[:, :, buttons].mean(dim=1) > 0.5, chunked[:, :, button_count : button_count + 2].sum(dim=1)


# CEMis good enough for now
@torch.no_grad()
def search(
    objective: WorldModelObjective,
    clip: dict[str, Tensor],
    buttons: Tensor,
    prior: Tensor,
    settings: ActionSearchConfig,
    modality: CostModality,
    generator: torch.Generator,
) -> dict[str, Tensor]:
    layout = ControlLayout()
    video, audio, actions = clip["video"][None], clip["audio"][None], clip["actions"]
    context_steps = objective.context_steps
    chunks = (actions.shape[0] - context_steps) // settings.chunk_steps
    normalized_video = objective.normalizer.normalize_video(video)
    normalized_audio = objective.prepare_audio(objective.normalizer.normalize_audio(audio))
    noise_video = torch.randn(
        (1, actions.shape[0] - context_steps, *video.shape[2:]), generator=generator, device=video.device
    )
    noise_audio = torch.randn(
        (1, actions.shape[0] - context_steps, normalized_audio.shape[-1]), generator=generator, device=video.device
    )
    probabilities = repeat(prior, "button -> chunk button", chunk=chunks).clone()
    mean = torch.zeros(chunks, 2, device=video.device)
    deviation = torch.full((chunks, 2), settings.axis_deviation, device=video.device)
    context = actions[:context_steps]
    best = {"cost": torch.tensor(float("inf"), device=video.device)}
    for iteration in range(settings.iterations):
        keys = (
            torch.rand((settings.population, *probabilities.shape), generator=generator, device=video.device)
            < probabilities
        ).float()
        axes = mean + deviation * torch.randn(
            (settings.population, chunks, 2), generator=generator, device=video.device
        )
        controls = candidate_controls(context, keys, axes, buttons, layout, settings.chunk_steps)
        costs = torch.cat(
            [
                plan_cost(
                    objective,
                    predict(
                        objective,
                        normalized_video,
                        normalized_audio,
                        group,
                        noise_video,
                        noise_audio,
                        settings.integration_steps,
                    ),
                    normalized_video,
                    normalized_audio,
                    context_steps,
                    modality,
                )
                for group in controls.split(settings.candidate_batch)
            ]
        )
        elite = costs.topk(settings.elites, largest=False).indices
        if costs[elite[0]] < best["cost"]:
            best = {"cost": costs[elite[0]], "controls": controls[elite[0], context_steps:]}
        probabilities = settings.momentum * probabilities + (1 - settings.momentum) * keys[elite].mean(dim=0)
        mean = settings.momentum * mean + (1 - settings.momentum) * axes[elite].mean(dim=0)
        deviation = settings.momentum * deviation + (1 - settings.momentum) * axes[elite].std(dim=0).clamp_min(0.05)
    recorded = actions[None]
    true_cost = plan_cost(
        objective,
        predict(
            objective,
            normalized_video,
            normalized_audio,
            recorded,
            noise_video,
            noise_audio,
            settings.integration_steps,
        ),
        normalized_video,
        normalized_audio,
        context_steps,
        modality,
    )[0]
    held = repeat(actions[context_steps - 1], "control -> time control", time=actions.shape[0] - context_steps)
    return {
        "inferred": best["controls"],
        "recorded": actions[context_steps:],
        "held": held,
        "inferred_cost": best["cost"],
        "recorded_cost": true_cost,
    }


def control_scores(
    results: list[dict[str, Tensor]], buttons: Tensor, chunk_steps: int, minimum_motion: float
) -> dict[str, float]:
    layout = ControlLayout()
    scores: dict[str, float] = {}
    for guess in ("inferred", "held"):
        positives = predicted_positives = true_positives = 0.0
        cosines = []
        for result in results:
            true_keys, true_motion = chunk_summary(result["recorded"], buttons, layout, chunk_steps)
            keys, motion = chunk_summary(result[guess], buttons, layout, chunk_steps)
            true_positives += float((keys & true_keys).sum())
            predicted_positives += float(keys.sum())
            positives += float(true_keys.sum())
            moving = true_motion.norm(dim=-1) > minimum_motion
            if moving.any():
                cosines.append(torch.nn.functional.cosine_similarity(motion[moving], true_motion[moving], dim=-1))
        scores[f"{guess}/key_f1"] = 2 * true_positives / max(predicted_positives + positives, 1.0)
        scores[f"{guess}/mouse_cosine"] = float(torch.cat(cosines).mean()) if cosines else 0.0
    scores["inferred/cost_ratio"] = float(
        torch.stack([result["inferred_cost"] / result["recorded_cost"] for result in results]).mean()
    )
    return scores


@torch.no_grad()
def main(run: Path, checkpoint: Path, cache: Path, settings: ActionSearchConfig) -> None:
    config = PretrainConfig.model_validate_json((run / "config.json").read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bfloat16 = config.trainer.precision == "bf16"
    encoder_config = config.encoder.resolve(bfloat16=bfloat16)
    objective = restore_objective(config, encoder_config, checkpoint_weights(checkpoint, True), device)
    dataset = CachedClipDataset(cache)
    clips = [dataset[index] for index in range(len(dataset))]
    steps = config.encoder.total_frames // encoder_config.video_frames_per_step
    clips = [
        {
            name: tensor[-kept_steps(name, steps, objective.audio_history_steps) :] if tensor.dim() else tensor
            for name, tensor in clip.items()
        }
        for clip in clips
        if int(clip["action_present"]) == 1
    ][: settings.clips]
    layout = ControlLayout()
    button_count = layout.keyboard_count + len(layout.mouse_buttons)
    pressed = torch.stack([clip["actions"][:, :button_count] for clip in clips]).mean(dim=(0, 1))
    buttons = torch.nonzero(pressed > settings.minimum_press_rate).flatten().to(device)
    prior = pressed[buttons.cpu()].to(device)
    modalities: tuple[CostModality, ...]
    match config.objective.mode:
        case "video_only":
            modalities = ("video",)
        case "audio_only":
            modalities = ("audio",)
        case "audio_visual":
            modalities = ("video", "audio", "audio_video")
    summary: dict[str, float] = {"clips": float(len(clips)), "buttons": float(buttons.numel())}
    for modality in modalities:
        generator = torch.Generator(device=device).manual_seed(settings.seed)
        results = []
        for index, clip in enumerate(clips):
            clip = {name: tensor.to(device).float() for name, tensor in clip.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bfloat16):
                results.append(search(objective, clip, buttons, prior, settings, modality, generator))
            if index % 16 == 0:
                print(f"{modality} clip {index}/{len(clips)}", flush=True)
        scores = control_scores(results, buttons, settings.chunk_steps, settings.minimum_motion)
        summary.update({f"{modality}/{name}": value for name, value in scores.items()})
    for name, value in summary.items():
        print(f"{name}: {value:.4f}", flush=True)
    save_summary(summary, checkpoint, "action_inference", cache)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("checkpoint")
    parser.add_argument("cache", type=Path)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    settings = ActionSearchConfig.model_validate(dict(override.split("=", 1) for override in args.overrides))
    main(args.run, resolve_checkpoint(args.run, args.checkpoint), args.cache, settings)
