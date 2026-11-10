from einops import rearrange, repeat
from jaxtyping import Float
from safetensors import safe_open
from torch import Tensor
from transformers import AutoConfig, PeAudioEncoder
from models.audio.base import AudioEncoder
from models.encoder_types import FrozenEncoderConfig


class PeAvAudioEncoder(AudioEncoder):
    weight_prefix = "audio_model.audio_encoder."

    def __init__(self, config: FrozenEncoderConfig) -> None:
        super().__init__(config)
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
        self.hop_length = encoder_config.dac_config.hop_length
        assert encoder_config.hidden_size == config.audio_feature_dimension
        assert encoder_config.dac_config.sampling_rate == config.sample_rate
        assert config.sample_rate == config.audio_native_frame_rate * self.hop_length

    def features(self, mono: Float[Tensor, "mono sample"]) -> Float[Tensor, "mono feature time"]:
        config = self.config
        assert mono.shape[-1] % self.hop_length == 0
        features = self.encoder(input_values=rearrange(mono, "mono sample -> mono 1 sample")).last_hidden_state
        assert features.shape[1] == mono.shape[-1] // self.hop_length
        return repeat(features, "mono native feature -> mono feature (native repeat)", repeat=config.audio_frame_repeat)
