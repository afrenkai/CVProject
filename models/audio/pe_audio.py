from pathlib import Path
import torch
from beartype import beartype
from einops import rearrange, repeat
from jaxtyping import Float, jaxtyped
from safetensors import safe_open
from torch import Tensor, nn
from transformers import AutoConfig, PeAudioEncoder
from models.encoder_types import FrozenEncoderConfig


class PeAvAudioEncoder(nn.Module):
    sample_rate = 48000
    weight_prefix = "audio_model.audio_encoder."

    @staticmethod
    def feature_dimension(directory: Path) -> int:
        return AutoConfig.from_pretrained(
            str(directory), local_files_only=True
        ).audio_video_config.audio_config.hidden_size

    def __init__(self, config: FrozenEncoderConfig, device: torch.device) -> None:
        super().__init__()
        assert config.sample_rate == self.sample_rate
        assert config.audio_native_frame_rate == 25 and config.audio_frame_repeat == 2
        self.settings = config
        directory = config.audio_model_directory
        encoder_config = AutoConfig.from_pretrained(
            str(directory), local_files_only=True
        ).audio_video_config.audio_config
        self.encoder = PeAudioEncoder(encoder_config)
        with safe_open(str(directory / "model.safetensors"), "pt") as handle:
            weights = {
                name.removeprefix(self.weight_prefix): handle.get_tensor(name)
                for name in handle.keys()  # noqa: SIM118
                if name.startswith(self.weight_prefix)
            }
        self.encoder.load_state_dict(weights, strict=True)
        self.encoder = self.encoder.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
        assert encoder_config.hidden_size == config.audio_feature_dimension
        assert encoder_config.dac_config.sampling_rate == config.sample_rate
        self.hop_length = encoder_config.dac_config.hop_length
        assert config.sample_rate == config.audio_native_frame_rate * self.hop_length

    @torch.no_grad()
    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        waveform: Float[Tensor, "batch channel sample"],
    ) -> Float[Tensor, "mono_batch feature time"]:
        config = self.settings
        assert waveform.shape[1] == 2 and torch.isfinite(waveform).all()
        assert waveform.shape[-1] % self.hop_length == 0
        mono = rearrange(waveform, "batch channel sample -> (batch channel) 1 sample")
        features = self.encoder(input_values=mono.float()).last_hidden_state
        native_frames = waveform.shape[-1] // self.hop_length
        assert features.shape == (
            mono.shape[0],
            native_frames,
            config.audio_feature_dimension,
        )
        assert torch.isfinite(features).all()
        return repeat(
            features,
            "mono native feature -> mono feature (native repeat)",
            repeat=config.audio_frame_repeat,
        )
