from pathlib import Path
from typing import Self
import torch
from safetensors.torch import load_model
from torch import Tensor
from transformers import PreTrainedModel
from transformers.utils import cached_file
from evals.samples import rollout
from models.encoder_types import EncoderConfig, FrozenEncoderConfig
from models.joint_encoder import JointEncoder
from models.model_types import WorldModelConfig
from models.normalization import AudioVisualNormalizer
from training.objective import WorldModelObjective, build_objective
from training.training_types import ObjectiveConfig
from utils.hub_types import SpeakUpConfig


class SpeakUp(PreTrainedModel):
    config_class = SpeakUpConfig

    def __init__(self, config: SpeakUpConfig) -> None:
        super().__init__(config)
        settings = EncoderConfig.model_validate(config.encoder)
        self.encoder_config: FrozenEncoderConfig = settings.resolve(bfloat16=config.encoder_bfloat16)
        normalizer = AudioVisualNormalizer(
            torch.zeros(self.encoder_config.video_latent_dimension),
            torch.ones(self.encoder_config.video_latent_dimension),
            torch.zeros(self.encoder_config.audio_latent_dimension),
            torch.ones(self.encoder_config.audio_latent_dimension),
        )
        self.objective: WorldModelObjective = build_objective(
            self.encoder_config,
            settings.total_frames,
            WorldModelConfig.model_validate(config.model),
            ObjectiveConfig.model_validate(config.objective),
            normalizer,
            torch.device("cpu"),
        )
        self.encoder = JointEncoder(self.encoder_config, torch.device("cpu"))

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str | Path, *model_arguments: object, **settings: object
    ) -> Self:
        location = str(pretrained_model_name_or_path)
        config = settings.get("config")
        model = cls(config if isinstance(config, SpeakUpConfig) else SpeakUpConfig.from_pretrained(location))
        weights = cached_file(location, "model.safetensors")
        assert weights is not None, f"no model.safetensors in {location}"
        load_model(model.objective, weights)
        return model.eval()

    @torch.no_grad()
    def encode(self, frames: Tensor, waveform: Tensor) -> tuple[Tensor, Tensor]:
        return self.encoder.encode(frames, waveform)

    @torch.no_grad()
    def rollout(
        self,
        video: Tensor,
        audio: Tensor,
        actions: Tensor,
        action_present: Tensor,
        integration_steps: int = 8,
        seed: int = 0,
    ) -> dict[str, Tensor]:
        batch = {"video": video, "audio": audio, "actions": actions, "action_present": action_present}
        return rollout(self.objective, batch, integration_steps, seed)

    def forward(self, video: Tensor, audio: Tensor, actions: Tensor, action_present: Tensor) -> dict[str, Tensor]:
        return self.objective(video, audio, actions, action_present)
