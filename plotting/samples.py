import matplotlib
import torch
from einops import rearrange
from matplotlib import pyplot
from matplotlib.figure import Figure
from torch import Tensor
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram
from models.encoder_types import FrozenEncoderConfig
from training.samples import horizon_errors, sweep_metrics

matplotlib.use("Agg")


def error_figure(errors: dict[int, dict[str, Tensor]], seconds_per_step: float) -> Figure:
    figure, axes = pyplot.subplots(1, 2, figsize=(12, 8), constrained_layout=True)
    reference = errors[max(errors)]
    horizon = torch.arange(1, reference["video"].shape[0] + 1) * seconds_per_step
    for column, modality in enumerate(("video", "audio")):
        for steps, values in errors.items():
            axes[column].plot(horizon, values[modality].cpu(), label=f"model, {steps} NFE")
        axes[column].plot(
            horizon, reference[f"{modality}_persistence"].cpu(), label="copy last step", linestyle="--", color="black"
        )
        axes[column].set(title=f"{modality} latent MSE", xlabel="horizon (s)")
        axes[column].legend()
    return figure


def spread_figure(curves: dict[str, dict[str, Tensor]], seconds_per_step: float) -> Figure:
    figure, axes = pyplot.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for column, modality in enumerate(("video", "audio")):
        for label, values in curves.items():
            spread = values[f"{modality}_spread"].cpu()
            horizon = torch.arange(1, spread.shape[0] + 1) * seconds_per_step
            axes[column].plot(horizon, spread, label=label)
        axes[column].axhline(1.0, linestyle="--", color="black", label="matches true variance")
        axes[column].set(title=f"{modality} spread (predicted std / true std)", xlabel="horizon (s)", ylim=(0, 1.3))
        axes[column].legend()
    return figure


def pca_rgb(latents: Tensor, basis: Tensor, mean: Tensor, low: Tensor, high: Tensor, grid: int) -> Tensor:
    projected = (latents - mean) @ basis
    scaled = ((projected - low) / (high - low)).clamp(0.0, 1.0)
    return rearrange(scaled, "time (row column) channel -> time row column channel", row=grid)


def video_figure(
    result: dict[str, Tensor], frames: Tensor | None, encoder_config: FrozenEncoderConfig, columns: int = 6
) -> Figure:
    true_video, predicted_video = result["true_video"][0].float(), result["predicted_video"][0].float()
    target_steps, patches = true_video.shape[:2]
    grid = int(patches**0.5)
    flattened = rearrange(true_video, "time patch feature -> (time patch) feature")
    mean = flattened.mean(dim=0)
    basis = torch.pca_lowrank(flattened - mean, q=3, center=False)[2]
    projected = (flattened - mean) @ basis
    low, high = projected.quantile(0.01, dim=0), projected.quantile(0.99, dim=0)
    true_rgb = pca_rgb(true_video, basis, mean, low, high, grid)
    predicted_rgb = pca_rgb(predicted_video, basis, mean, low, high, grid)
    steps = torch.linspace(0, target_steps - 1, columns).round().long()
    labels = ("true latent", "predicted latent") if frames is None else ("frame", "true latent", "predicted latent")
    figure, axes = pyplot.subplots(
        len(labels), columns, figsize=(2 * columns, 2.1 * len(labels)), constrained_layout=True, squeeze=False
    )
    offset = len(labels) - 2
    for column, step in enumerate(steps.tolist()):
        frame_index = encoder_config.context_frames + step * encoder_config.video_frames_per_step
        seconds = (step + 1) * encoder_config.video_frames_per_step / encoder_config.video_rate
        if frames is not None:
            axes[0, column].imshow(
                rearrange(frames[0, frame_index].float().cpu(), "channel height width -> height width channel")
            )
        axes[offset, column].imshow(true_rgb[step].cpu())
        axes[offset + 1, column].imshow(predicted_rgb[step].cpu())
        axes[0, column].set_title(f"+{seconds:.1f}s")
    for row, label in enumerate(labels):
        axes[row, 0].set_ylabel(label)
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])
    return figure


def audio_figure(
    result: dict[str, Tensor], waveform: Tensor | None, encoder_config: FrozenEncoderConfig, components: int = 8
) -> Figure:
    context, true_future = result["context_audio"][0].float(), result["true_audio"][0].float()
    predicted_future = result["predicted_audio"][0].float()
    true_sequence = torch.cat((context, true_future))
    mean = true_sequence.mean(dim=0)
    basis = torch.pca_lowrank(true_sequence - mean, q=components, center=False)[2]
    true_projection = (true_sequence - mean) @ basis
    predicted_projection = (torch.cat((context, predicted_future)) - mean) @ basis
    limit = true_projection.abs().quantile(0.99).item()
    boundary = context.shape[0] - 0.5
    rows = 2 if waveform is None else 3
    figure, axes = pyplot.subplots(rows, 1, figsize=(10, 2.2 * rows), constrained_layout=True)
    if waveform is not None:
        spectrogram = AmplitudeToDB()(
            MelSpectrogram(encoder_config.loading_sample_rate, n_fft=2048, hop_length=480, n_mels=96)(
                waveform[0, 0].float().cpu()
            )
        )
        seconds = waveform.shape[-1] / encoder_config.loading_sample_rate
        axes[0].imshow(spectrogram, aspect="auto", origin="lower", extent=(0, seconds, 0, 96), cmap="magma")
        axes[0].axvline(encoder_config.context_frames / encoder_config.video_rate, color="white", linestyle="--")
        axes[0].set(title="input mel spectrogram (context | future)", xlabel="seconds", ylabel="mel bin")
    panels = {"true": true_projection, "predicted": predicted_projection}
    for index, label in enumerate(panels):
        axis, projection = axes[rows - 2 + index], panels[label]
        axis.imshow(projection.T.cpu(), aspect="auto", origin="lower", cmap="coolwarm", vmin=-limit, vmax=limit)
        axis.axvline(boundary, color="black", linestyle="--")
        axis.set(title=f"{label} audio latent, top {components} PCs (context | future)", xlabel="step", ylabel="PC")
    return figure


def sample_report(
    results: dict[int, dict[str, Tensor]],
    frames: Tensor | None,
    waveform: Tensor | None,
    encoder_config: FrozenEncoderConfig,
) -> tuple[dict[str, float], dict[str, Figure]]:
    errors = {steps: horizon_errors(result) for steps, result in results.items()}
    finest = results[max(results)]
    seconds_per_step = encoder_config.video_frames_per_step / encoder_config.video_rate
    return sweep_metrics(errors), {
        "video": video_figure(finest, frames, encoder_config),
        "horizon_error": error_figure(errors, seconds_per_step),
        "horizon_spread": spread_figure({f"{steps} NFE": values for steps, values in errors.items()}, seconds_per_step),
        "audio": audio_figure(finest, waveform, encoder_config),
    }
