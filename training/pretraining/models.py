import torch
from omegaconf import DictConfig, OmegaConf
from architecture.av_model import AudioVisualFlowPredictor
from architecture.pretraining.audio.audio_decoder import CausalAudioDecoder
from architecture.pretraining.audio.encoder import HybridAudioEncoder
from architecture.pretraining.latent_actions import LatentActionModel
from architecture.pretraining.normalization import AudioVisualNormalizer
from architecture.pretraining.recorded_controls import RecordedControlEncoder
from architecture.pretraining.types import WorldModelConfig
from architecture.pretraining.video.encoder import DinoVideoEncoder
from architecture.pretraining.video.video_decoder import DinoVideoDecoder
from train_datasets.control_types import ControlEncoderConfig
from training.pretraining.types import PretrainingModels


def build_flow_predictor(
    video_dimension: int,
    audio_dimension: int,
    config: WorldModelConfig,
    device: torch.device,
) -> AudioVisualFlowPredictor:
    return AudioVisualFlowPredictor(
        video_dimension,
        audio_dimension,
        config.action_dimension,
        config.backbone.hidden_dimension,
        config.backbone.heads,
        config.backbone.layers,
        config.diffusion_head.hidden_dimension,
        config.diffusion_head.heads,
        config.diffusion_head.layers,
        config.mlp_ratio,
        config.dropout,
        config.backbone_attention,
        config.hybrid_history,
        config.noise_time_embedding,
        config.spatial_grid,
        config.audio_tokens,
    ).to(device)


def build_pretraining_models(
    config: DictConfig, device: torch.device, pretrained_video: bool | None = None
) -> PretrainingModels:
    world_model = WorldModelConfig.model_validate(OmegaConf.to_container(config.world_model, resolve=True))
    video_encoder = DinoVideoEncoder(
        config.video_encoder.model_id,
        config.video_encoder.pretrained if pretrained_video is None else pretrained_video,
        config.video_encoder.image_size,
        config.video_encoder.layer_numbers,
        config.video_encoder.mean,
        config.video_encoder.standard_deviation,
    ).to(device)
    audio_encoder = HybridAudioEncoder(
        config.audio_encoder.input_channels,
        config.audio_encoder.hidden_dimension,
        config.audio_encoder.latent_dimension,
        config.audio_encoder.downsample_strides,
        config.audio_encoder.block_pattern,
        config.audio_encoder.kda_heads,
        config.audio_encoder.kda_head_dimension,
        config.audio_encoder.kda_chunk_size,
        config.audio_encoder.kda_minimum_log_decay,
        config.audio_encoder.attention_heads,
        config.audio_encoder.attention_window_steps,
    ).to(device)
    audio_decoder = CausalAudioDecoder(
        config.audio_encoder.latent_dimension,
        config.audio_encoder.input_channels,
        audio_encoder.total_stride,
    ).to(device)
    video_decoder = DinoVideoDecoder(
        config.video_encoder.feature_dimension,
        config.video_decoder.hidden_dimension,
        config.video_decoder.upsample_channels,
        config.video_decoder.residual_blocks,
        config.video_decoder.normalization_groups,
        config.video_decoder.patch_grid_size,
    ).to(device)
    normalizer = AudioVisualNormalizer(
        torch.zeros(config.video_encoder.feature_dimension),
        torch.ones(config.video_encoder.feature_dimension),
        torch.zeros(config.audio_encoder.latent_dimension),
        torch.ones(config.audio_encoder.latent_dimension),
    ).to(device)
    latent_action_model = LatentActionModel(
        config.video_encoder.feature_dimension,
        config.audio_encoder.latent_dimension,
        config.latent_actions.hidden_dimension,
        world_model.action_dimension,
        config.latent_actions.code_count,
        config.latent_actions.ema_decay,
        config.latent_actions.ema_epsilon,
        config.world_pretraining.get("latent_action_audio", "transition"),
    ).to(device)
    predictor = build_flow_predictor(
        config.video_encoder.feature_dimension,
        config.audio_encoder.latent_dimension,
        world_model,
        device,
    )
    controls = config.world_pretraining.get("recorded_controls")
    control_encoder = (
        RecordedControlEncoder(
            ControlEncoderConfig.model_validate(OmegaConf.to_container(controls, resolve=True)),
            world_model.action_dimension,
        ).to(device)
        if controls is not None
        else None
    )
    return PretrainingModels(
        video_encoder=video_encoder,
        audio_encoder=audio_encoder,
        audio_decoder=audio_decoder,
        video_decoder=video_decoder,
        normalizer=normalizer,
        latent_action_model=latent_action_model,
        predictor=predictor,
        control_encoder=control_encoder,
    )
