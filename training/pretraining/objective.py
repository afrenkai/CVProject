import torch
from beartype import beartype
from einops import rearrange
from jaxtyping import Float, Int, jaxtyped
from torch import Tensor, nn
from architecture.av_model import AudioVisualFlowPredictor
from architecture.pretraining.encoder_types import EncodeFeatures
from architecture.pretraining.normalization import AudioVisualNormalizer
from architecture.pretraining.recorded_controls import RecordedControlEncoder
from losses.flow import flow_matching_terms
from losses.types import FlowLossConfig
from train_datasets.control_types import ControlEncoderConfig
from transport.flow import AudioVisualFlowTransport


class PretrainingObjective(nn.Module):
    def __init__(
        self,
        predictor: AudioVisualFlowPredictor,
        transport: AudioVisualFlowTransport,
        context_steps: int,
        action_config: ControlEncoderConfig,
        normalizer: AudioVisualNormalizer,
        encode_features: EncodeFeatures,
        loss_config: FlowLossConfig,
    ) -> None:
        super().__init__()
        self.predictor = predictor
        self.transport = transport
        self.context_steps = context_steps
        self.control_encoder = RecordedControlEncoder(action_config, predictor.null_action.shape[0])
        self.normalizer = normalizer
        self.encode_features = encode_features
        self.loss_config = loss_config
        self.training_loss_sums: dict[str, Tensor] = {}
        self.training_sample_count = 0

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
    def feature_loss(
        self,
        video: Float[Tensor, "batch time patch video_feature"],
        audio: Float[Tensor, "batch time audio_feature"],
        actions: Float[Tensor, "batch time control"],
        action_present: Int[Tensor, "batch"],  # noqa: F821
        example_seed: Int[Tensor, "batch"],  # noqa: F821
        return_loss: bool = True,
    ) -> dict[str, Tensor]:
        conditioned_actions = self.encode_actions(actions, action_present)
        devices = [video.device.index] if video.is_cuda else []
        with torch.random.fork_rng(devices=devices, enabled=not self.training):
            if not self.training:
                torch.manual_seed(int(example_seed.sum()) + 10000)
            terms = flow_matching_terms(
                self.transport,
                self.predictor,
                video,
                audio,
                self.context_steps,
                conditioned_actions[:, self.context_steps :],
                video_loss_weight=self.loss_config.video_weight,
                audio_loss_weight=self.loss_config.audio_weight,
                temporal_loss_weight=self.loss_config.temporal_weight,
                context_actions=conditioned_actions[:, : self.context_steps],
            )
        assert return_loss and terms.total.dtype == torch.float32 and torch.isfinite(terms.total)
        components = terms.components()
        if self.training:
            for name, value in components.items():
                increment = value.detach() * video.shape[0]
                self.training_loss_sums[name] = self.training_loss_sums.get(name, value.new_zeros(())) + increment
            self.training_sample_count += video.shape[0]
        return {"loss": terms.total} | {
            name: rearrange(value.detach().expand(video.shape[0]), "batch -> batch 1")
            for name, value in components.items()
        }

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        frames: Float[Tensor, "batch frame video_channel height width"],
        waveform: Float[Tensor, "batch audio_channel sample"],
        actions: Float[Tensor, "batch time control"],
        action_present: Int[Tensor, "batch"],  # noqa: F821
        example_seed: Int[Tensor, "batch"],  # noqa: F821
        return_loss: bool = True,
    ) -> dict[str, Tensor]:
        with torch.no_grad():
            video, audio = self.encode_features(frames, waveform)
            video = self.normalizer.normalize_video(video)
            audio = self.normalizer.normalize_audio(audio)
        assert video.dtype == audio.dtype == torch.float32
        return self.feature_loss(video, audio, actions, action_present, example_seed, return_loss)
