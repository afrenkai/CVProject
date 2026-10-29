import math
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from beartype import beartype
from einops import rearrange
from jaxtyping import Float, jaxtyped
from torch import Tensor
from architecture.pretraining.encoder_types import FrozenEncoderConfig
from evaluation.evaluation_types import ValidationResult


@torch.no_grad()
@jaxtyped(typechecker=beartype)
def render_video_mean(
    reference: Float[Tensor, "time patch feature"],
    prediction: Float[Tensor, "time patch feature"],
    destination: Path,
    block_seconds: float,
    seed: int,
    n_futures: int,
) -> None:
    height = math.isqrt(reference.shape[1])
    assert height * height == reference.shape[1]
    flattened = rearrange(reference, "time patch feature -> (time patch) feature")
    center = flattened.mean(dim=0)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        projection = torch.pca_lowrank(flattened - center, q=3, center=False)[2]
    reference_rgb, prediction_rgb = (reference - center) @ projection, (prediction - center) @ projection
    low = reference_rgb.amin(dim=(0, 1))
    scale = (reference_rgb.amax(dim=(0, 1)) - low).clamp_min(0.000001)
    times = torch.linspace(0, reference.shape[0] - 1, min(8, reference.shape[0])).long().tolist()
    figure, axes = plt.subplots(2, len(times), figsize=(2 * len(times), 4), squeeze=False)
    for column, frame in enumerate(times):
        for row, values in enumerate((reference_rgb, prediction_rgb)):
            rgb = ((values[frame] - low) / scale).clamp(0, 1)
            axes[row, column].imshow(
                rearrange(rgb, "(height width) color -> height width color", height=height).numpy()
            )
            axes[row, column].set_title(f"+{(frame + 1) * block_seconds:.1f}s")
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
    axes[0, 0].set_ylabel("Target")
    axes[1, 0].set_ylabel(f"{n_futures}-future mean")
    figure.suptitle("Normalized video latent PCA; shared target basis, no decoded video")
    figure.tight_layout()
    figure.savefig(destination, dpi=130)
    plt.close(figure)


@torch.no_grad()
@jaxtyped(typechecker=beartype)
def render_audio_mean(
    reference: Float[Tensor, "time feature"],
    prediction: Float[Tensor, "time feature"],
    destination: Path,
    tokens_per_step: int,
    future_seconds: float,
    n_futures: int,
) -> None:
    target = rearrange(
        reference, "time (channel phase feature) -> channel (time phase) feature", channel=2, phase=tokens_per_step
    )
    mean = rearrange(
        prediction, "time (channel phase feature) -> channel (time phase) feature", channel=2, phase=tokens_per_step
    )
    columns = torch.linspace(0, target.shape[-1] - 1, min(256, target.shape[-1])).long()
    images = (target[..., columns], mean[..., columns], (mean[..., columns] - target[..., columns]).abs())
    bound = max(float(target.abs().quantile(0.99)), 0.000001)
    figure, axes = plt.subplots(3, 2, figsize=(14, 8))
    for row, values in enumerate(images):
        for channel in range(2):
            axes[row, channel].imshow(
                values[channel].numpy().T,
                aspect="auto",
                origin="lower",
                extent=(0, future_seconds, 0, len(columns)),
                cmap="magma" if row == 2 else "coolwarm",
                vmin=0 if row == 2 else -bound,
                vmax=bound,
            )
            axes[row, channel].set_ylabel(("Target", f"{n_futures}-future mean", "Absolute error")[row])
            axes[row, channel].set_title(f"Channel {channel + 1}")
    axes[2, 0].set_xlabel("Future seconds")
    axes[2, 1].set_xlabel("Future seconds")
    figure.suptitle("Normalized Dasheng features; sampled columns, no generated waveform")
    figure.tight_layout()
    figure.savefig(destination, dpi=130)
    plt.close(figure)


@torch.no_grad()
@jaxtyped(typechecker=beartype)
def render_sample_mean(
    video_mean: Float[Tensor, "one future patch video_feature"],
    audio_mean: Float[Tensor, "one future audio_feature"],
    *,
    directory: Path,
    video: Float[Tensor, "one time patch video_feature"],
    audio: Float[Tensor, "one time audio_feature"],
    context: int,
    config: FrozenEncoderConfig,
    seed: int,
    n_futures: int,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    block_seconds = config.video_frames_per_step / config.video_rate
    render_video_mean(
        video[0, context:].float().cpu(),
        video_mean[0].float().cpu(),
        directory / "video_latent_pca.png",
        block_seconds,
        seed,
        n_futures,
    )
    render_audio_mean(
        audio[0, context:].float().cpu(),
        audio_mean[0].float().cpu(),
        directory / "audio_latents.png",
        config.audio_tokens_per_step,
        audio_mean.shape[1] * block_seconds,
        n_futures,
    )


def render_validation_summary(directory: Path, result: ValidationResult) -> None:
    corpora = sorted(result.per_corpus)
    metrics = {
        "Video sample MSE": "video/mean_sample_mse",
        "Audio sample MSE": "audio/mean_sample_mse",
        "Flow loss": "flow/unweighted_flow_loss",
    }
    figure, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4), squeeze=False)
    for column, (title, key) in enumerate(metrics.items()):
        axes[0, column].bar(corpora, [result.per_corpus[corpus].mean.get(key, float("nan")) for corpus in corpora])
        axes[0, column].set_title(title)
        axes[0, column].tick_params(axis="x", rotation=45)
    figure.suptitle(
        f"Pretraining validation, step {result.global_step}: {result.evaluated_windows} windows, "
        f"{result.n_futures} futures, {result.integration_steps} solver steps"
    )
    figure.tight_layout()
    figure.savefig(directory / "review.png", dpi=130)
    plt.close(figure)
