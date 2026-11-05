import torch
from huggingface_hub import PyTorchModelHubMixin
from torch import nn
from models.encoder_types import EncoderConfig, FrozenEncoderConfig
from models.joint_encoder import JointEncoder, build_encoder_config
from models.model_types import WorldModelConfig
from models.normalization import AudioVisualNormalizer
from training.objective import WorldModelObjective, build_objective
from training.training_types import ObjectiveConfig


class SpeakUp(
    nn.Module,
    PyTorchModelHubMixin,
    library_name="speakup",
    tags=["world-model", "audio-visual", "flow-matching"],
):
    def __init__(self, config: dict[str, dict[str, object]]) -> None:
        super().__init__()
        self.encoder_settings = EncoderConfig.model_validate(config["encoder"])
        self.encoder_config: FrozenEncoderConfig = build_encoder_config(self.encoder_settings, bfloat16=False)
        normalizer = AudioVisualNormalizer(
            torch.zeros(self.encoder_config.video_latent_dimension),
            torch.ones(self.encoder_config.video_latent_dimension),
            torch.zeros(self.encoder_config.audio_latent_dimension),
            torch.ones(self.encoder_config.audio_latent_dimension),
        )
        self.objective: WorldModelObjective = build_objective(
            self.encoder_config,
            self.encoder_settings.total_frames,
            WorldModelConfig.model_validate(config["model"]),
            ObjectiveConfig.model_validate(config["objective"]),
            normalizer,
            torch.device("cpu"),
        )

    def load_encoder(self, device: torch.device) -> JointEncoder:
        return JointEncoder(self.encoder_config, device).eval()
