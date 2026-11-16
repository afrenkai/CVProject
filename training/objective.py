import math
from pathlib import Path
import torch
from beartype import beartype
from einops import rearrange, reduce
from jaxtyping import Float, Int, jaxtyped
from safetensors.torch import load_file
from torch import Tensor, nn
from data.control_types import ControlEncoderConfig, ControlLayout
from losses.flow import flow_matching_terms
from losses.slim_lsd import slim_lsd_loss
from models.encoder_types import FrozenEncoderConfig
from models.model_types import AudioTokenConfig, WorldModelConfig
from models.normalization import AudioVisualNormalizer
from models.recorded_controls import RecordedControlEncoder
from models.world_model import AudioVisualFlowPredictor
from training.training_types import (
    ModalityDropoutConfig,
    ObjectiveConfig,
    ObjectiveMode,
    PretrainConfig,
    SlimLsdConfig,
)
from transport.flow import AudioVisualFlowTransport
from transport.transport_types import TimeDistributionConfig, TransportConfig


class WorldModelObjective(nn.Module):
    def __init__(
        self,
        predictor: AudioVisualFlowPredictor,
        transport: AudioVisualFlowTransport,
        context_steps: int,
        control_encoder: RecordedControlEncoder,
        normalizer: AudioVisualNormalizer,
        mode: ObjectiveMode,
        slim_lsd: SlimLsdConfig,
        audio_layout: AudioTokenConfig,
        audio_step_pooling: bool,
        spatial_audio_target: bool,
    ) -> None:
        super().__init__()
        self.predictor = predictor
        self.transport = transport
        self.context_steps = context_steps
        self.control_encoder = control_encoder
        self.normalizer = normalizer
        self.mode = mode
        self.video_weight = 0.0 if mode == "audio_only" else 1.0
        self.audio_weight = 0.0 if mode == "video_only" else 1.0
        self.audio_layout = audio_layout
        self.audio_step_pooling = audio_step_pooling
        self.scored_channels = audio_layout.channels if spatial_audio_target else min(2, audio_layout.channels)
        channel_mask = (torch.arange(audio_layout.channels) < self.scored_channels).float()
        token_features = predictor.audio_tokens.parts * predictor.backbone_target_tokens.audio_projection.in_features
        self.register_buffer(
            "audio_mask",
            rearrange(
                channel_mask[:, None, None].expand(-1, predictor.audio_tokens.frames_per_step, token_features),
                "channel phase feature -> (channel phase feature)",
            ).contiguous(),
            persistent=False,
        )
        self.slim_lsd = slim_lsd
        self.slim_lsd_weight = 0.0
        self.diagonal_fraction = 1.0
        self.action_dropout = 0.0
        self.modality_dropout = ModalityDropoutConfig()
        self.audio_history_steps = 0

    @jaxtyped(typechecker=beartype)
    def encode_actions(
        self,
        actions: Float[Tensor, "batch time control"],
        action_present: Int[Tensor, "batch"],  # noqa: F821
    ) -> Float[Tensor, "batch time action_feature"]:
        encoded = self.control_encoder(actions)
        null = self.predictor.null_action.to(encoded.dtype).expand_as(encoded)
        return torch.where(action_present[:, None, None].bool(), encoded, null)

    def prepare_audio(self, audio: Tensor) -> Tensor:
        if not self.audio_step_pooling:
            return audio
        return reduce(
            audio,
            "... (channel phase feature) -> ... (channel feature)",
            "mean",
            channel=self.audio_layout.channels,
            phase=self.audio_layout.frames_per_step,
        )

    def scored_audio(self, audio: Tensor) -> Tensor:
        channels = rearrange(audio, "... (channel rest) -> ... channel rest", channel=self.audio_layout.channels)
        return rearrange(channels[..., : self.scored_channels, :], "... channel rest -> ... (channel rest)")

    def silence(self, video: Tensor, audio: Tensor) -> tuple[Tensor, Tensor]:
        match self.mode:
            case "video_only":
                return video, torch.zeros_like(audio)
            case "audio_only":
                return torch.zeros_like(video), audio
            case "audio_visual":
                return video, audio

    @jaxtyped(typechecker=beartype)
    def drop_context(
        self,
        video: Float[Tensor, "batch time patch video_feature"],
        audio: Float[Tensor, "batch audio_time audio_feature"],
    ) -> tuple[Float[Tensor, "batch time patch video_feature"], Float[Tensor, "batch audio_time audio_feature"]]:
        dropout = self.modality_dropout
        if not self.training or dropout.audio_context + dropout.video_context <= 0:
            return video, audio
        draw = torch.rand(video.shape[0], device=video.device)
        audio_dropped = draw < dropout.audio_context
        video_dropped = (draw >= dropout.audio_context) & (draw < dropout.audio_context + dropout.video_context)
        video_steps = torch.arange(video.shape[1], device=video.device) < self.context_steps
        audio_steps = torch.arange(audio.shape[1], device=audio.device) < self.context_steps + self.audio_history_steps
        audio_keep = ~(audio_dropped[:, None] & audio_steps[None, :])
        video_keep = ~(video_dropped[:, None] & video_steps[None, :])
        return video * video_keep[:, :, None, None], audio * audio_keep[:, :, None]

    @jaxtyped(typechecker=beartype)
    def flow_loss(
        self,
        video: Float[Tensor, "batch time patch video_feature"],
        audio: Float[Tensor, "batch audio_time audio_feature"],
        actions: Float[Tensor, "batch time control"],
        action_present: Int[Tensor, "batch"],  # noqa: F821
    ) -> dict[str, Tensor]:
        if self.training and self.action_dropout > 0:
            keep = torch.rand(action_present.shape[0], device=video.device) >= self.action_dropout
            action_present = action_present * keep
        conditioned = self.encode_actions(actions, action_present)
        loss, video_loss, audio_loss = flow_matching_terms(
            self.transport,
            self.predictor,
            video,
            audio,
            self.context_steps,
            conditioned[:, self.context_steps :],
            conditioned[:, : self.context_steps],
            self.video_weight,
            self.audio_weight,
            self.audio_mask,
        )
        return {"loss": loss, "video_flow_loss": video_loss.detach(), "audio_flow_loss": audio_loss.detach()}

    def forward(self, video: Tensor, audio: Tensor, actions: Tensor, action_present: Tensor) -> dict[str, Tensor]:
        video, audio = self.drop_context(*self.silence(video, self.prepare_audio(audio)))
        if self.slim_lsd_weight <= 0 or self.diagonal_fraction >= 1.0:
            return self.flow_loss(video, audio, actions, action_present)
        batch = video.shape[0]
        diagonal = min(batch, int(self.diagonal_fraction * batch + float(torch.rand(()))))
        outputs = {"loss": video.new_zeros((), dtype=torch.float32)}
        if diagonal > 0:
            flow = self.flow_loss(video[:diagonal], audio[:diagonal], actions[:diagonal], action_present[:diagonal])
            outputs = {**flow, "loss": flow["loss"] * diagonal / batch}
        if diagonal == batch:
            return outputs
        conditioned = self.encode_actions(actions[diagonal:], action_present[diagonal:])
        distillation, residual = slim_lsd_loss(
            self.transport,
            self.predictor,
            video[diagonal:],
            audio[diagonal:],
            self.context_steps,
            conditioned[:, self.context_steps :],
            conditioned[:, : self.context_steps],
            self.slim_lsd.maximum_gap,
            self.slim_lsd.maximum_end_time,
            self.slim_lsd.adaptive_power,
        )
        return {
            **outputs,
            "loss": outputs["loss"] + self.slim_lsd_weight * distillation * (batch - diagonal) / batch,
            "slim_lsd_loss": residual,
        }


def build_objective(
    encoder: FrozenEncoderConfig,
    total_frames: int,
    model: WorldModelConfig,
    objective: ObjectiveConfig,
    normalizer: AudioVisualNormalizer,
    device: torch.device,
) -> WorldModelObjective:
    audio_layout = AudioTokenConfig(
        channels=encoder.audio_channels,
        frames_per_step=encoder.audio_tokens_per_step,
        parts=encoder.audio_token_parts,
    )
    pooled = objective.audio_step_pooling
    audio_tokens = audio_layout.model_copy(update={"frames_per_step": 1}) if pooled else audio_layout
    audio_dimension = encoder.audio_latent_dimension // (encoder.audio_tokens_per_step if pooled else 1)
    predictor = AudioVisualFlowPredictor(encoder.video_latent_dimension, audio_dimension, audio_tokens, model)
    patches = encoder.video_patches
    target_steps = (total_frames - encoder.context_frames) // encoder.video_frames_per_step
    data_dimension = target_steps * (patches * encoder.video_latent_dimension + audio_dimension)
    transport = TransportConfig(
        time_distribution=TimeDistributionConfig(shift=objective.time_shift or math.sqrt(data_dimension / 4096)),
        context_noise=objective.context_noise,
    )
    layout = ControlLayout()
    control_encoder = RecordedControlEncoder(
        ControlEncoderConfig(
            layout=layout,
            axis_scales=(1.0,) * len(layout.axis_names),
            hidden_dimension=128,
        ),
        predictor.null_action.shape[0],
    )
    world_model = WorldModelObjective(
        predictor,
        AudioVisualFlowTransport(transport),
        encoder.context_frames // encoder.video_frames_per_step,
        control_encoder,
        normalizer,
        objective.mode,
        objective.slim_lsd,
        audio_layout,
        objective.audio_step_pooling,
        objective.spatial_audio_target,
    ).to(device=device, dtype=torch.float32)
    world_model.action_dropout = objective.action_dropout
    world_model.modality_dropout = objective.modality_dropout
    world_model.audio_history_steps = encoder.kept_audio_history_steps
    return world_model


def checkpoint_weights(checkpoint: Path, use_ema: bool) -> dict[str, Tensor]:
    if not use_ema:
        return load_file(checkpoint / "model.safetensors")
    averaged = torch.load(checkpoint / "ema.pt", map_location="cpu", weights_only=False)
    return {name.removeprefix("module."): tensor for name, tensor in averaged.items() if name != "n_averaged"}


def restore_objective(
    config: PretrainConfig, encoder: FrozenEncoderConfig, weights: dict[str, Tensor], device: torch.device
) -> WorldModelObjective:
    video, audio = encoder.video_latent_dimension, encoder.audio_latent_dimension
    normalizer = AudioVisualNormalizer(torch.zeros(video), torch.ones(video), torch.zeros(audio), torch.ones(audio))
    objective = build_objective(
        encoder, config.encoder.total_frames, config.model, config.objective, normalizer, device
    )
    objective.load_state_dict(weights)
    return objective.eval()
