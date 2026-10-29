import tempfile
from pathlib import Path
import fire
import torch
from architecture.pretraining.encoder_types import FrozenEncoderConfig
from architecture.pretraining.model_types import WorldModelConfig
from architecture.pretraining.normalization import AudioVisualNormalizer
from evaluation.likelihood import bits_per_dimension, flow_negative_log_likelihood
from train_datasets.control_types import ControlLayout
from training.simple_objective import build_latent_objective
from transport.transport_types import TransportConfig


def main(
    checkpoint: str,
    cache: str,
    windows: int = 16,
    steps: tuple[int, ...] = (16, 32, 128, 512),
    probes: int = 2,
    device: str = "cuda",
) -> None:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    data = torch.load(cache)
    assert state["encoder"] == data["encoder"]
    normalizer_state = state["normalizer"]
    normalizer = AudioVisualNormalizer(
        normalizer_state["video.mean"],
        normalizer_state["video.standard_deviation"],
        normalizer_state["audio.mean"],
        normalizer_state["audio.standard_deviation"],
    )
    encoder = FrozenEncoderConfig.model_validate(state["encoder"])
    with tempfile.TemporaryDirectory() as scratch:
        objective = build_latent_objective(
            encoder,
            WorldModelConfig.model_validate(state["world_model"]),
            TransportConfig.model_validate(state["transport"]),
            normalizer,
            1.0,
            Path(scratch),
            torch.device(device),
        )
    objective.load_state_dict(state["objective"])
    objective.eval()
    video = normalizer.normalize_video(data["video"][:windows].float()).to(device)
    audio = normalizer.normalize_audio(data["audio"][:windows].float()).to(device)
    context = objective.context_steps
    actions = torch.zeros(windows, video.shape[1], ControlLayout().dimension, device=device)
    conditioned = objective.encode_actions(actions, torch.zeros(windows, dtype=torch.int64, device=device))
    target_video = objective.transport.video_target(video[:, context:], video[:, :context])
    for step_count in steps:
        grid = torch.linspace(0.0, 1.0, step_count + 1, device=device)
        for probe in range(probes):
            with torch.no_grad():
                video_nll, audio_nll = flow_negative_log_likelihood(
                    objective.predictor,
                    target_video,
                    audio[:, context:],
                    video[:, :context],
                    audio[:, :context],
                    conditioned[:, context:],
                    grid,
                    objective.transport.config.noise_standard_deviation,
                    torch.Generator().manual_seed(probe),
                    context_actions=conditioned[:, :context],
                )
            print(
                f"steps {step_count:4d} probe {probe}: video {bits_per_dimension(video_nll, target_video).mean():10.4f}"
                f"  audio {bits_per_dimension(audio_nll, audio[:, context:]).mean():10.4f} bits/dim",
                flush=True,
            )


if __name__ == "__main__":
    fire.Fire(main)
