import json
import math
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
import fire
import torch
from einops import rearrange, reduce
from torch import Tensor, nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from architecture.pretraining.encoder_types import FrozenEncoderConfig
from architecture.pretraining.model_types import AudioTokenConfig, TransformerStageConfig, WorldModelConfig
from architecture.pretraining.normalization import AudioVisualNormalizer
from evaluation.likelihood import bits_per_dimension, flow_negative_log_likelihood
from losses.flow import flow_matching_terms
from train_datasets.control_types import ControlLayout
from training.pretraining.objective import PretrainingObjective
from training.simple_objective import build_latent_objective, feature_moments
from transport.solvers import solve_flow_map, solve_flow_map_given_audio
from transport.transport_types import TransportConfig


def modality_gradient_norms(
    objective: PretrainingObjective, video: Tensor, audio: Tensor, actions: Tensor, action_present: Tensor
) -> dict[str, float]:
    context = objective.context_steps
    conditioned = objective.encode_actions(actions, action_present)
    terms = flow_matching_terms(
        objective.transport,
        objective.predictor,
        video,
        audio,
        context,
        conditioned[:, context:],
        context_actions=conditioned[:, :context],
    )
    shared = [parameter for parameter in objective.predictor.backbone_blocks.parameters() if parameter.requires_grad]
    norms = {}
    for name, loss in (("video", terms.video_flow), ("audio", terms.audio_flow)):
        gradients = torch.autograd.grad(loss, shared, retain_graph=True, allow_unused=True)
        norms[name] = float(sum(gradient.square().sum() for gradient in gradients if gradient is not None) ** 0.5)
    return norms


def train(
    objective: PretrainingObjective,
    video: Tensor,
    audio: Tensor,
    steps: int,
    batch_size: int,
    learning_rate: float,
    log_every: int,
    label: str,
    lr_schedule: str,
    warmup_steps: int,
    ema_decay: float,
) -> PretrainingObjective:
    time_steps = video.shape[1]
    actions = torch.zeros(batch_size, time_steps, ControlLayout().dimension, device=video.device)
    action_present = torch.zeros(batch_size, dtype=torch.int64, device=video.device)
    optimizer = torch.optim.AdamW(objective.parameters(), lr=learning_rate)
    scheduler = None
    if lr_schedule == "cosine" and steps > warmup_steps:
        scheduler = SequentialLR(
            optimizer,
            [
                LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps),
                CosineAnnealingLR(optimizer, T_max=steps - warmup_steps, eta_min=0.05 * learning_rate),
            ],
            milestones=[warmup_steps],
        )
    averaged = (
        AveragedModel(objective, multi_avg_fn=get_ema_multi_avg_fn(ema_decay), use_buffers=True) if ema_decay else None
    )
    objective.train()
    running = {"video_flow_loss": 0.0, "audio_flow_loss": 0.0}
    started = time.perf_counter()
    for step in range(1, steps + 1):
        batch_index = torch.randint(video.shape[0], (batch_size,), device=video.device)
        outputs = objective.feature_loss(
            video[batch_index].float(),
            audio[batch_index].float(),
            actions,
            action_present,
            batch_index,
        )
        optimizer.zero_grad()
        outputs["loss"].backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if averaged is not None:
            averaged.update_parameters(objective)
        for name in running:
            running[name] += outputs[name].mean().item()
        if step % log_every == 0:
            losses = "  ".join(f"{name} {value / log_every:.4f}" for name, value in running.items())
            norms = modality_gradient_norms(
                objective, video[batch_index].float(), audio[batch_index].float(), actions, action_present
            )
            gradients = f"backbone grad norm video {norms['video']:.3e} audio {norms['audio']:.3e}"
            print(f"[{label}] step {step}: {losses}  {gradients}  ({time.perf_counter() - started:.0f}s)", flush=True)
            running = dict.fromkeys(running, 0.0)
    objective.eval()
    if averaged is None:
        return objective
    averaged.module.eval()
    return averaged.module


def window_errors(prediction: Tensor, target: Tensor, moving_mask: Tensor) -> tuple[Tensor, Tensor]:
    squared = reduce((prediction - target).square(), "window time patch feature -> window patch", "mean")
    whole = squared.mean(dim=1)
    moving = (squared * moving_mask).sum(dim=1) / moving_mask.sum(dim=1)
    return whole, moving


def fit_side_probe(
    target_video: Tensor, sides: Tensor, seed: int, probe_features: str, steps: int = 500
) -> Callable[[Tensor], Tensor]:
    generator = torch.Generator().manual_seed(seed)
    projection = (torch.randn(target_video.shape[-1], 32, generator=generator) / target_video.shape[-1] ** 0.5).to(
        target_video.device
    )

    def features(latents: Tensor) -> Tensor:
        selected = latents[:, -1] - latents[:, 0] if probe_features == "change" else latents[:, -1]
        return rearrange(selected.float() @ projection, "window patch reduced -> window (patch reduced)")

    inputs = features(target_video)
    mean, scale = inputs.mean(dim=0), inputs.std(dim=0).clamp_min(1e-6)
    classifier = nn.Linear(inputs.shape[1], 1).to(target_video.device)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=1e-2, weight_decay=1e-2)
    labels = sides.float().to(target_video.device)
    with torch.enable_grad():
        for step in range(steps):
            loss = nn.functional.binary_cross_entropy_with_logits(
                classifier((inputs - mean) / scale).squeeze(-1), labels
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def predict(latents: Tensor) -> Tensor:
        with torch.no_grad():
            return (classifier((features(latents) - mean) / scale).squeeze(-1) > 0).long()

    return predict


@torch.no_grad()
def evaluate(
    objectives: dict[str, PretrainingObjective],
    video: Tensor,
    audio: Tensor,
    futures: int,
    integration_steps: int,
    batch_size: int,
    seed: int,
    likelihood_steps: int,
    side_probe: Callable[[Tensor], Tensor] | None = None,
    sides: Tensor | None = None,
) -> dict[str, Tensor]:
    audio_visual = objectives["audio_visual"]
    context = audio_visual.context_steps
    partner = (torch.arange(video.shape[0]) + video.shape[0] // 2) % video.shape[0]
    errors: dict[str, list[Tensor]] = {}

    def record(name: str, values: Tensor) -> None:
        errors.setdefault(name, []).append(values)

    started = time.perf_counter()
    for batch_start in range(0, video.shape[0], batch_size):
        index = torch.arange(batch_start, min(batch_start + batch_size, video.shape[0]))
        batch_video, batch_audio = video[index].float(), audio[index].float()
        context_video, target_video = batch_video[:, :context], batch_video[:, context:]
        context_audio, target_audio = batch_audio[:, :context], batch_audio[:, context:]
        persistence_video = context_video[:, -1:].expand_as(target_video)
        motion = reduce(
            (target_video - persistence_video).square(), "window time patch feature -> window patch", "mean"
        )
        moving_mask = torch.zeros_like(motion)
        moving_mask.scatter_(1, motion.topk(math.ceil(motion.shape[1] / 4), dim=1).indices, 1.0)

        actions = torch.zeros(index.shape[0], video.shape[1], ControlLayout().dimension, device=video.device)
        action_present = torch.zeros(index.shape[0], dtype=torch.int64, device=video.device)
        conditioned = {
            name: objective.encode_actions(actions, action_present) for name, objective in objectives.items()
        }
        grid = audio_visual.transport.time_grid(integration_steps, batch_video)
        context_audio_variants = {
            "matched_audio_context": context_audio,
            "mean_audio_context": torch.zeros_like(context_audio),
            "swapped_audio_context": audio[partner[index], :context].float(),
        }

        whole, moving = window_errors(persistence_video, target_video, moving_mask)
        record("persistence/video_whole", whole)
        record("persistence/video_moving", moving)
        record(
            "persistence/audio",
            reduce((context_audio[:, -1:] - target_audio).square(), "window time feature -> window", "mean"),
        )
        if side_probe is not None:
            record("side_error/true_future", (side_probe(target_video) != sides[index]).float())

        per_future: dict[str, list[Tensor]] = {}
        for future in range(futures):
            generator = torch.Generator().manual_seed(seed * 1_000_003 + batch_start * 101 + future)
            video_noise = torch.randn(target_video.shape, generator=generator).to(video.device)
            audio_noise = torch.randn(target_audio.shape, generator=generator).to(video.device)

            for variant, variant_context_audio in context_audio_variants.items():
                sampled_video, sampled_audio = solve_flow_map(
                    audio_visual.predictor,
                    video_noise,
                    audio_noise,
                    context_video,
                    variant_context_audio,
                    conditioned["audio_visual"][:, context:],
                    grid,
                    diagonal=True,
                    context_actions=conditioned["audio_visual"][:, :context],
                )
                sampled_video = audio_visual.transport.video_frames(sampled_video, context_video)
                whole, moving = window_errors(sampled_video, target_video, moving_mask)
                per_future.setdefault(f"audio_visual/{variant}/video_whole", []).append(whole)
                per_future.setdefault(f"audio_visual/{variant}/video_moving", []).append(moving)
                if side_probe is not None:
                    per_future.setdefault(f"side_error/audio_visual/{variant}", []).append(
                        (side_probe(sampled_video) != sides[index]).float()
                    )
                per_future.setdefault(f"audio_visual/{variant}/audio", []).append(
                    reduce((sampled_audio - target_audio).square(), "window time feature -> window", "mean")
                )

            given_audio_video = solve_flow_map_given_audio(
                audio_visual.predictor,
                video_noise,
                audio_noise,
                target_audio,
                context_video,
                context_audio,
                conditioned["audio_visual"][:, context:],
                grid,
                context_actions=conditioned["audio_visual"][:, :context],
            )
            given_audio_video = audio_visual.transport.video_frames(given_audio_video, context_video)
            whole, moving = window_errors(given_audio_video, target_video, moving_mask)
            per_future.setdefault("audio_visual/true_future_audio/video_whole", []).append(whole)
            per_future.setdefault("audio_visual/true_future_audio/video_moving", []).append(moving)
            if side_probe is not None:
                per_future.setdefault("side_error/audio_visual/true_future_audio", []).append(
                    (side_probe(given_audio_video) != sides[index]).float()
                )

            video_only = objectives["video_only"]
            sampled_video, sampled_audio = solve_flow_map(
                video_only.predictor,
                video_noise,
                audio_noise,
                context_video,
                torch.zeros_like(context_audio),
                conditioned["video_only"][:, context:],
                grid,
                diagonal=True,
                context_actions=conditioned["video_only"][:, :context],
            )
            sampled_video = video_only.transport.video_frames(sampled_video, context_video)
            whole, moving = window_errors(sampled_video, target_video, moving_mask)
            per_future.setdefault("video_only/video_whole", []).append(whole)
            per_future.setdefault("video_only/video_moving", []).append(moving)
            if side_probe is not None:
                per_future.setdefault("side_error/video_only", []).append(
                    (side_probe(sampled_video) != sides[index]).float()
                )

        if likelihood_steps > 0:
            likelihood_grid = torch.linspace(0.0, 1.0, likelihood_steps + 1, device=video.device)
            noise_scale = audio_visual.transport.config.noise_standard_deviation
            for variant, variant_context_audio in context_audio_variants.items():
                generator = torch.Generator().manual_seed(seed * 7_000_003 + batch_start)
                video_nll, audio_nll = flow_negative_log_likelihood(
                    audio_visual.predictor,
                    audio_visual.transport.video_target(target_video, context_video),
                    target_audio,
                    context_video,
                    variant_context_audio,
                    conditioned["audio_visual"][:, context:],
                    likelihood_grid,
                    noise_scale,
                    generator,
                    context_actions=conditioned["audio_visual"][:, :context],
                )
                record(f"likelihood/audio_visual/{variant}/video_bits", bits_per_dimension(video_nll, target_video))
                record(f"likelihood/audio_visual/{variant}/audio_bits", bits_per_dimension(audio_nll, target_audio))
            generator = torch.Generator().manual_seed(seed * 7_000_003 + batch_start)
            video_nll, audio_nll = flow_negative_log_likelihood(
                objectives["video_only"].predictor,
                objectives["video_only"].transport.video_target(target_video, context_video),
                torch.zeros_like(target_audio),
                context_video,
                torch.zeros_like(context_audio),
                conditioned["video_only"][:, context:],
                likelihood_grid,
                noise_scale,
                generator,
                context_actions=conditioned["video_only"][:, :context],
            )
            record("likelihood/video_only/video_bits", bits_per_dimension(video_nll, target_video))

        for name, values in per_future.items():
            record(name, torch.stack(values).mean(dim=0))
        done = int(index[-1]) + 1
        print(f"evaluated {done}/{video.shape[0]} windows ({time.perf_counter() - started:.0f}s)", flush=True)
    return {name: torch.cat(values).cpu() for name, values in errors.items()}


def describe(values: Tensor) -> str:
    standard_error = values.std() / math.sqrt(values.shape[0])
    return f"{values.mean():.4f} ± {standard_error:.4f}"


def report(errors: dict[str, Tensor], keep: Tensor, title: str) -> dict[str, dict[str, float]]:
    print(f"\n== {title}: {int(keep.sum())} windows (mean ± standard error across windows)")
    for name in sorted(errors):
        print(f"  {name:55s} {describe(errors[name][keep])}")
    comparisons = {
        "video_whole: audio_visual matched vs video_only": (
            "audio_visual/matched_audio_context/video_whole",
            "video_only/video_whole",
        ),
        "video_moving: audio_visual matched vs video_only": (
            "audio_visual/matched_audio_context/video_moving",
            "video_only/video_moving",
        ),
        "video_whole: true future audio vs joint": (
            "audio_visual/true_future_audio/video_whole",
            "audio_visual/matched_audio_context/video_whole",
        ),
        "video_moving: true future audio vs joint": (
            "audio_visual/true_future_audio/video_moving",
            "audio_visual/matched_audio_context/video_moving",
        ),
        "video_moving: matched vs mean audio context": (
            "audio_visual/matched_audio_context/video_moving",
            "audio_visual/mean_audio_context/video_moving",
        ),
        "video_moving: matched vs swapped audio context": (
            "audio_visual/matched_audio_context/video_moving",
            "audio_visual/swapped_audio_context/video_moving",
        ),
        "video bits/dim: matched vs swapped audio context": (
            "likelihood/audio_visual/matched_audio_context/video_bits",
            "likelihood/audio_visual/swapped_audio_context/video_bits",
        ),
        "audio bits/dim: matched vs swapped audio context": (
            "likelihood/audio_visual/matched_audio_context/audio_bits",
            "likelihood/audio_visual/swapped_audio_context/audio_bits",
        ),
        "side error: audio_visual matched vs video_only": (
            "side_error/audio_visual/matched_audio_context",
            "side_error/video_only",
        ),
        "side error: matched vs swapped audio context": (
            "side_error/audio_visual/matched_audio_context",
            "side_error/audio_visual/swapped_audio_context",
        ),
        "video bits/dim: audio_visual matched vs video_only": (
            "likelihood/audio_visual/matched_audio_context/video_bits",
            "likelihood/video_only/video_bits",
        ),
    }
    comparisons = {label: names for label, names in comparisons.items() if names[0] in errors and names[1] in errors}
    print("  paired differences (negative = first is better), fraction of windows where first is better:")
    summary = {}
    for label, (first, second) in comparisons.items():
        difference = errors[first][keep] - errors[second][keep]
        better = float((difference < 0).float().mean())
        print(f"  {label:55s} {describe(difference)}  better on {better:.0%}")
        summary[label] = {
            "mean": float(difference.mean()),
            "standard_error": float(difference.std() / math.sqrt(difference.shape[0])),
            "fraction_better": better,
        }
    return summary


def main(
    train_cache: str,
    eval_cache: str,
    output: str,
    steps: int = 3000,
    batch_size: int = 16,
    learning_rate: float = 3e-4,
    speech_quantile: float = 0.4,
    hidden_dimension: int = 512,
    heads: int = 8,
    layers: int = 4,
    head_hidden_dimension: int = 256,
    head_heads: int = 8,
    head_layers: int = 2,
    prediction: str = "data",
    audio_token_channels: int = 2,
    audio_token_frames: int = 5,
    key_value_heads: int = 0,
    head_key_value_heads: int = 0,
    attention_window: int = 0,
    futures: int = 4,
    integration_steps: int = 16,
    likelihood_steps: int = 32,
    eval_batch_size: int = 8,
    log_every: int = 100,
    seed: int = 0,
    device: str = "cpu",
    video_target: str = "change",
    init_audio_visual: str = "",
    init_video_only: str = "",
    normalizer_source: str = "checkpoint",
    train_examples: int = 0,
    probe_features: str = "final",
    lr_schedule: str = "constant",
    warmup_steps: int = 500,
    ema_decay: float = 0.0,
) -> None:
    assert lr_schedule in {"constant", "cosine"}
    assert probe_features in {"final", "change"}
    assert normalizer_source in {"checkpoint", "train"}
    initial = {
        label: torch.load(path, map_location="cpu", weights_only=False)
        for label, path in (("audio_visual", init_audio_visual), ("video_only", init_video_only))
        if path
    }
    train_data = torch.load(train_cache)
    eval_data = torch.load(eval_cache)
    assert train_data["encoder"] == eval_data["encoder"]
    encoder_cfg = FrozenEncoderConfig.model_validate(train_data["encoder"])
    threshold = torch.quantile(train_data["target_rms"], speech_quantile)
    train_keep = train_data["target_rms"] >= threshold
    train_index = train_keep.nonzero().flatten()
    if train_examples:
        train_index = train_index[:train_examples]
    if train_index.shape[0] == train_keep.shape[0]:
        train_video, train_audio = train_data["video"], train_data["audio"]
    else:
        train_video, train_audio = train_data["video"][train_index], train_data["audio"][train_index]
    print(
        f"Train: {train_index.shape[0]}/{train_keep.shape[0]} windows (audio threshold, then first {train_examples or 'all'})"
    )
    print(f"Eval: {eval_data['video'].shape[0]} windows")
    if "actor" in train_data:
        print(f"Actors: train {train_data['actor'].unique().tolist()}, eval {eval_data['actor'].unique().tolist()}")

    if initial:
        assert set(initial) == {"audio_visual", "video_only"}
        assert all(checkpoint["encoder"] == train_data["encoder"] for checkpoint in initial.values())
    if initial and normalizer_source == "checkpoint":
        normalizer_state = initial["audio_visual"]["normalizer"]
        normalizer = AudioVisualNormalizer(
            normalizer_state["video.mean"],
            normalizer_state["video.standard_deviation"],
            normalizer_state["audio.mean"],
            normalizer_state["audio.standard_deviation"],
        )
    else:
        video_mean, video_std = feature_moments(train_video, (0, 1, 2))
        audio_mean, audio_std = feature_moments(train_audio, (0, 1))
        normalizer = AudioVisualNormalizer(video_mean, video_std, audio_mean, audio_std)

    def normalized(video: Tensor, audio: Tensor) -> tuple[Tensor, Tensor]:
        for chunk in video.split(64):
            chunk.copy_(normalizer.normalize_video(chunk.float()))
        for chunk in audio.split(64):
            chunk.copy_(normalizer.normalize_audio(chunk.float()))
        return video, audio

    train_video, train_audio = normalized(train_video, train_audio)
    eval_video, eval_audio = normalized(eval_data["video"], eval_data["audio"])
    train_video, train_audio = train_video.to(device), train_audio.to(device)
    eval_video, eval_audio = eval_video.to(device), eval_audio.to(device)

    world_model = WorldModelConfig(
        action_dimension=64,
        backbone_attention="quadratic",
        backbone=TransformerStageConfig(
            hidden_dimension=hidden_dimension, heads=heads, layers=layers, key_value_heads=key_value_heads or None
        ),
        diffusion_head=TransformerStageConfig(
            hidden_dimension=head_hidden_dimension,
            heads=head_heads,
            layers=head_layers,
            key_value_heads=head_key_value_heads or None,
        ),
        prediction=prediction,
        attention_window=attention_window or None,
        audio_tokens=AudioTokenConfig(channels=audio_token_channels, frames_per_step=audio_token_frames),
    )
    transport = TransportConfig(video_target=video_target)
    if initial:
        world_model = WorldModelConfig.model_validate(initial["audio_visual"]["world_model"])
        transport = TransportConfig.model_validate(initial["audio_visual"]["transport"])
    print(f"World model: {world_model.model_dump()}")
    objectives = {}
    with tempfile.TemporaryDirectory() as scratch:
        for label, audio_weight, audio_input in (
            ("audio_visual", 1.0, train_audio),
            ("video_only", 0.0, torch.zeros_like(train_audio)),
        ):
            torch.manual_seed(seed)
            objectives[label] = build_latent_objective(
                encoder_cfg, world_model, transport, normalizer, audio_weight, Path(scratch), torch.device(device)
            )
            if label in initial:
                objectives[label].load_state_dict(initial[label]["objective"])
            torch.manual_seed(seed + 1)
            objectives[label] = train(
                objectives[label],
                train_video,
                audio_input,
                steps,
                batch_size,
                learning_rate,
                log_every,
                label,
                lr_schedule,
                warmup_steps,
                ema_decay,
            )

    side_probe, sides = None, None
    if "side" in train_data and "side" in eval_data:
        context_steps = encoder_cfg.context_frames // encoder_cfg.video_frames_per_step
        side_probe = fit_side_probe(
            train_video[:, context_steps:], train_data["side"][train_index], seed, probe_features
        )
        sides = eval_data["side"].to(device)
    errors = evaluate(
        objectives,
        eval_video,
        eval_audio,
        futures,
        integration_steps,
        eval_batch_size,
        seed,
        likelihood_steps,
        side_probe,
        sides,
    )
    speech = eval_data["target_rms"] >= threshold
    results = {
        "config": {
            "steps": steps,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "speech_quantile": speech_quantile,
            "world_model": world_model.model_dump(),
            "transport": transport.model_dump(),
            "init_audio_visual": init_audio_visual,
            "init_video_only": init_video_only,
            "futures": futures,
            "integration_steps": integration_steps,
            "seed": seed,
            "train_windows": int(train_index.shape[0]),
            "normalizer_source": normalizer_source if initial else "train",
            "lr_schedule": lr_schedule,
            "warmup_steps": warmup_steps,
            "ema_decay": ema_decay,
            "eval_windows": int(speech.shape[0]),
            "eval_speech_windows": int(speech.sum()),
        },
        "all_windows": report(errors, torch.ones_like(speech), "Held-out windows, all"),
        "speech_windows": report(errors, speech, "Held-out windows above the train audio threshold"),
        "per_window": {name: values.tolist() for name, values in errors.items()},
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(results, indent=2))
    print(f"\nSaved {output}")


if __name__ == "__main__":
    fire.Fire(main)
