import torch
import yaml
from torch import Tensor
from architecture.pretraining.encoder_types import EncodeFeatures
from architecture.pretraining.normalization import AudioVisualNormalizer
from train_datasets.control_types import ControlEncoderConfig
from training.pretraining.configuration import PretrainingConfig
from training.pretraining.models import build_flow_predictor
from training.pretraining.objective import PretrainingObjective
from transport.flow import AudioVisualFlowTransport


def load_normalizer(config: PretrainingConfig) -> AudioVisualNormalizer:
    video_dimension = config.encoder.video_feature_dimension * config.encoder.video_frames_per_step
    audio_dimension = config.encoder.audio_feature_dimension * 2 * config.encoder.audio_tokens_per_step
    normalizer = AudioVisualNormalizer(
        torch.zeros(video_dimension),
        torch.ones(video_dimension),
        torch.zeros(audio_dimension),
        torch.ones(audio_dimension),
    )
    state: dict[str, Tensor] = torch.load(config.normalization.latents, map_location="cpu", weights_only=True)
    normalizer.load_state_dict(state)
    assert all(value.dtype == torch.float32 and torch.isfinite(value).all() for value in normalizer.buffers())
    assert (normalizer.video.standard_deviation > 0).all() and (normalizer.audio.standard_deviation > 0).all()
    return normalizer


def build_online_objective(
    config: PretrainingConfig, encode_features: EncodeFeatures, device: torch.device
) -> PretrainingObjective:
    assert config.world_model.backbone_attention == "quadratic"
    video_dimension = config.encoder.video_feature_dimension * config.encoder.video_frames_per_step
    audio_dimension = config.encoder.audio_feature_dimension * 2 * config.encoder.audio_tokens_per_step
    predictor = build_flow_predictor(video_dimension, audio_dimension, config.world_model, device)
    predictor.outcome_head.requires_grad_(False)
    predictor.linear_mixer.requires_grad_(False)
    predictor.noise_time_projection.requires_grad_(False)
    predictor.activation_checkpointing = config.trainer.activation_checkpointing
    predictor.diffusion_head.activation_checkpointing = config.trainer.activation_checkpointing
    scales = yaml.safe_load(config.normalization.actions.read_text())
    assert scales["fit_split"] == "train"
    objective = PretrainingObjective(
        predictor,
        AudioVisualFlowTransport(config.transport),
        config.data.context_frames // config.data.video_frames_per_step,
        ControlEncoderConfig(
            layout=config.data.control_layout,
            axis_scales=tuple(scales["axis_scales"]),
            hidden_dimension=384,
            conditioning="recorded",
        ),
        load_normalizer(config),
        encode_features,
        config.loss,
    ).to(device=device, dtype=torch.float32)
    return objective
