import torch
from beartype import beartype
from einops import rearrange, repeat
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn
from transformers import AutoFeatureExtractor, AutoModel
from architecture.pretraining.encoder_types import FrozenEncoderConfig


class DashengAudioEncoder(nn.Module):
    def __init__(self, config: FrozenEncoderConfig, device: torch.device) -> None:
        super().__init__()
        assert config.sample_rate == 16000
        assert config.audio_native_frame_rate == 25 and config.audio_frame_repeat == 2
        self.settings = config
        self.processor = AutoFeatureExtractor.from_pretrained(
            str(config.audio_model_directory),
            trust_remote_code=True,
            local_files_only=True,
        )
        self.encoder = (
            AutoModel.from_pretrained(
                str(config.audio_model_directory),
                trust_remote_code=True,
                local_files_only=True,
                use_safetensors=True,
                outputdim=None,
            )
            .to(device=device, dtype=torch.float32)
            .eval()
            .requires_grad_(False)
        )
        assert self.encoder.config.name == "dasheng-1.2B"
        assert self.encoder.config.encoder_kwargs["embed_dim"] == config.audio_feature_dimension == 1536
        assert isinstance(self.encoder.outputlayer, nn.Identity)
        assert self.processor.sampling_rate == config.sample_rate
        assert self.processor.hop_size == 160 and self.processor.feature_size == 64

    @torch.no_grad()
    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        waveform: Float[Tensor, "batch channel sample"],
    ) -> Float[Tensor, "mono_batch feature time"]:
        config = self.settings
        assert waveform.shape[1] == 2 and torch.isfinite(waveform).all()
        assert waveform.shape[-1] * config.audio_native_frame_rate % config.sample_rate == 0
        mono = rearrange(waveform, "batch channel sample -> (batch channel) sample")
        spectrograms = [
            self.processor(
                channel_waveform.unsqueeze(0),
                sampling_rate=config.sample_rate,
                truncation=False,
                return_tensors="pt",
            ).input_values
            for channel_waveform in mono.unbind(dim=0)
        ]
        spectrogram = torch.cat(spectrograms)
        assert spectrogram.shape[-1] <= self.encoder.encoder.target_length
        features = self.encoder(input_values=spectrogram).hidden_states
        native_frames = waveform.shape[-1] * config.audio_native_frame_rate // config.sample_rate
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
