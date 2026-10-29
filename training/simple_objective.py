from pathlib import Path
import torch
import yaml
from pydantic import Field, PositiveInt
from torch import Tensor
from architecture.pretraining.encoder_types import EncodeFeatures, FrozenEncoderConfig
from architecture.pretraining.model_types import WorldModelConfig
from architecture.pretraining.normalization import AudioVisualNormalizer
from configs.config_types import FrozenModel
from losses.loss_types import FlowLossConfig
from train_datasets.control_types import ControlEncoderConfig, ControlLayout
from training.pretraining.models import build_flow_predictor
from training.pretraining.objective import PretrainingObjective
from transport.flow import AudioVisualFlowTransport
from transport.transport_types import TransportConfig


class SimpleObjectiveConfig(FrozenModel):

    normalizer: Path
    action_scales: Path
    encoder: FrozenEncoderConfig
    world_model: WorldModelConfig
    context_frames: PositiveInt = 60
    video_frames_per_step: PositiveInt = 3
    control_hidden_dimension: PositiveInt = 384
    transport: TransportConfig = Field(default_factory=TransportConfig)
    loss: FlowLossConfig = Field(default_factory=FlowLossConfig)
    control_layout: ControlLayout = Field(default_factory=ControlLayout)


def build_simple_objective(
    config: SimpleObjectiveConfig,
    encode_features: EncodeFeatures,
    device: torch.device,
) -> PretrainingObjective:
    video_dim = config.encoder.video_latent_dimension
    audio_dim = config.encoder.audio_latent_dimension

    predictor = build_flow_predictor(video_dim, audio_dim, config.world_model, device)

    normalizer = AudioVisualNormalizer(
        torch.zeros(video_dim),
        torch.ones(video_dim),
        torch.zeros(audio_dim),
        torch.ones(audio_dim),
    )
    state: dict[str, Tensor] = torch.load(config.normalizer, map_location="cpu", weights_only=True)
    normalizer.load_state_dict(state)
    assert all(v.dtype == torch.float32 and torch.isfinite(v).all() for v in normalizer.buffers())

    scales = yaml.safe_load(config.action_scales.read_text())
    action_config = ControlEncoderConfig(
        layout=config.control_layout,
        axis_scales=tuple(scales["axis_scales"]),
        hidden_dimension=config.control_hidden_dimension,
        conditioning="recorded",
    )

    return PretrainingObjective(
        predictor,
        AudioVisualFlowTransport(config.transport),
        config.context_frames // config.video_frames_per_step,
        action_config,
        normalizer,
        encode_features,
        config.loss,
    ).to(device=device, dtype=torch.float32)


def cached_latents_only(frames: Tensor, waveform: Tensor) -> tuple[Tensor, Tensor]:
    raise RuntimeError("This objective trains on latents encoded outside of it")


def feature_moments(latents: Tensor, reduced_dims: tuple[int, ...], chunk: int = 64) -> tuple[Tensor, Tensor]:
    total = torch.zeros(latents.shape[-1], dtype=torch.float64, device=latents.device)
    squared = torch.zeros(latents.shape[-1], dtype=torch.float64, device=latents.device)
    count = 0
    for start in range(0, latents.shape[0], chunk):
        values = latents[start : start + chunk].double()
        total += values.sum(dim=reduced_dims)
        squared += values.square().sum(dim=reduced_dims)
        count += values.numel() // values.shape[-1]
    mean = total / count
    return mean.float(), (squared / count - mean.square()).clamp_min(1e-8).sqrt().float()


def build_latent_objective(
    encoder: FrozenEncoderConfig,
    world_model: WorldModelConfig,
    transport: TransportConfig,
    normalizer: AudioVisualNormalizer,
    audio_weight: float,
    directory: Path,
    device: torch.device,
) -> PretrainingObjective:
    layout = ControlLayout()
    directory.mkdir(parents=True, exist_ok=True)
    normalizer_path = directory / "normalizer.pt"
    torch.save(normalizer.state_dict(), normalizer_path)
    scales_path = directory / "action_scales.yaml"
    scales_path.write_text(
        yaml.dump({"fit_split": "train", "fit_windows": 0, "axis_scales": [1.0] * len(layout.axis_names)})
    )
    return build_simple_objective(
        SimpleObjectiveConfig(
            normalizer=normalizer_path,
            action_scales=scales_path,
            encoder=encoder,
            world_model=world_model,
            context_frames=encoder.context_frames,
            video_frames_per_step=encoder.video_frames_per_step,
            control_hidden_dimension=128,
            transport=transport,
            loss=FlowLossConfig(audio_weight=audio_weight),
            control_layout=layout,
        ),
        cached_latents_only,
        device,
    )
