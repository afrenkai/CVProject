import torch
from beartype import beartype
from einops import rearrange, repeat
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram
from transformers import AutoFeatureExtractor, AutoModel
from models.encoder_types import FrozenEncoderConfig


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
        assert self.encoder.config.encoder_kwargs["embed_dim"] == config.audio_feature_dimension
        assert isinstance(self.encoder.outputlayer, nn.Identity)
        assert self.processor.sampling_rate == config.sample_rate
        assert self.processor.hop_size == 160 and self.processor.feature_size == 64
        self.spectrogram = MelSpectrogram(
            sample_rate=self.processor.sampling_rate,
            n_fft=self.processor.n_fft,
            win_length=self.processor.win_size,
            hop_length=self.processor.hop_size,
            f_min=self.processor.f_min,
            f_max=self.processor.f_max,
            n_mels=self.processor.feature_size,
            center=self.processor.center,
        ).to(device)
        self.decibels = AmplitudeToDB(top_db=120)

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
        spectrogram = rearrange(
            self.decibels(rearrange(self.spectrogram(mono.float()), "mono mel time -> mono 1 mel time")),
            "mono 1 mel time -> mono mel time",
        )
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
