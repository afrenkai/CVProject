import math
from pathlib import Path
import torch
from beartype import beartype
from jaxtyping import Float, Int, jaxtyped
from safetensors.torch import load_file
from torch import Tensor, nn
from data.control_types import ControlEncoderConfig, ControlLayout
from losses.flow import flow_matching_terms
from losses.loss_types import FlowLossConfig
from losses.slim_lsd import slim_lsd_loss
from models.encoder_types import FrozenEncoderConfig
from models.model_types import AudioTokenConfig, WorldModelConfig
from models.normalization import AudioVisualNormalizer
from models.recorded_controls import RecordedControlEncoder
from models.world_model import AudioVisualFlowPredictor
from training.training_types import ObjectiveConfig, PretrainConfig, SlimLsdConfig
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
        loss_config: FlowLossConfig,
        slim_lsd: SlimLsdConfig,
    ) -> None:
        super().__init__()
        self.predictor = predictor
        self.transport = transport
        self.context_steps = context_steps
        self.control_encoder = control_encoder
        self.normalizer = normalizer
        self.loss_config = loss_config
        self.slim_lsd = slim_lsd
        self.slim_lsd_weight = 0.0
        self.diagonal_fraction = 1.0
        self.action_dropout = 0.0
        self.context_dropout = 0.0

    @jaxtyped(typechecker=beartype)
    def encode_actions(
        self,
        actions: Float[Tensor, "batch time control"],
        action_present: Int[Tensor, "batch"],  # noqa: F821
    ) -> Float[Tensor, "batch time action_feature"]:
        encoded = self.control_encoder(actions)
        null = self.predictor.null_action.to(encoded.dtype).expand_as(encoded)
        return torch.where(action_present[:, None, None].bool(), encoded, null)

    @jaxtyped(typechecker=beartype)
    def flow_loss(
        self,
        video: Float[Tensor, "batch time patch video_feature"],
        audio: Float[Tensor, "batch time audio_feature"],
        actions: Float[Tensor, "batch time control"],
        action_present: Int[Tensor, "batch"],  # noqa: F821
    ) -> dict[str, Tensor]:
        rows = action_present.shape[0]
        context_keep = None
        if self.training and self.action_dropout > 0:
            action_present = action_present * (torch.rand(rows, device=video.device) >= self.action_dropout)
        conditioned = self.encode_actions(actions, action_present)
        context_actions = conditioned[:, : self.context_steps]
        if self.training and self.context_dropout > 0:
            context_keep = (torch.rand(rows, device=video.device) >= self.context_dropout).to(video.dtype)
            null = self.predictor.null_action.to(context_actions.dtype).expand_as(context_actions)
            context_actions = torch.where(context_keep[:, None, None].bool(), context_actions, null)
        terms = flow_matching_terms(
            self.transport,
            self.predictor,
            video,
            audio,
            self.context_steps,
            conditioned[:, self.context_steps :],
            video_loss_weight=self.loss_config.video_weight,
            audio_loss_weight=self.loss_config.audio_weight,
            temporal_loss_weight=self.loss_config.temporal_weight,
            context_actions=context_actions,
            context_keep=context_keep,
        )
        assert terms.total.dtype == torch.float32 and torch.isfinite(terms.total)
        return {
            "loss": terms.total,
            "video_flow_loss": terms.video_flow.detach(),
            "audio_flow_loss": terms.audio_flow.detach(),
        }

    def forward(self, video: Tensor, audio: Tensor, actions: Tensor, action_present: Tensor) -> dict[str, Tensor]:
        if self.slim_lsd_weight <= 0 or self.diagonal_fraction >= 1.0:
            return self.flow_loss(video, audio, actions, action_present)
        batch = video.shape[0]
        diagonal = min(batch, int(self.diagonal_fraction * batch + float(torch.rand(()))))
        outputs = {"loss": video.new_zeros((), dtype=torch.float32)}
        if diagonal > 0:
            flow = self.flow_loss(video[:diagonal], audio[:diagonal], actions[:diagonal], action_present[:diagonal])
            outputs = flow | {"loss": flow["loss"] * diagonal / batch}
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
            self.slim_lsd.maximum_gap,
            conditioned[:, : self.context_steps],
            self.slim_lsd.maximum_end_time,
            self.slim_lsd.adaptive_power,
        )
        return outputs | {
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
    model = model.model_copy(
        update={
            "audio_tokens": AudioTokenConfig(channels=encoder.audio_channels, frames_per_step=5),
            "context_noise_conditioning": objective.context_noise > 0,
        }
    )
    predictor = AudioVisualFlowPredictor(
        encoder.video_latent_dimension,
        encoder.audio_latent_dimension,
        model.action_dimension,
        model.backbone.hidden_dimension,
        model.backbone.heads,
        model.backbone.layers,
        model.diffusion_head.hidden_dimension,
        model.diffusion_head.heads,
        model.diffusion_head.layers,
        model.mlp_ratio,
        model.dropout,
        model.noise_time_embedding,
        model.spatial_grid,
        model.audio_tokens,
        model.prediction,
        model.backbone.key_value_heads,
        model.diffusion_head.key_value_heads,
        model.attention_window,
        model.attention_backend,
        model.context_noise_conditioning,
    )
    patches = (encoder.image_size // 16) ** 2
    target_steps = (total_frames - encoder.context_frames) // encoder.video_frames_per_step
    data_dimension = target_steps * (patches * encoder.video_latent_dimension + encoder.audio_latent_dimension)
    transport = TransportConfig(
        video_target=objective.video_target,
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
        FlowLossConfig(audio_weight=1.0 if objective.mode == "audio_visual" else 0.0),
        objective.slim_lsd,
    ).to(device=device, dtype=torch.float32)
    world_model.action_dropout = objective.action_dropout
    world_model.context_dropout = objective.context_dropout
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
