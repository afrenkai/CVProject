from einops import rearrange, repeat
from jaxtyping import Float
from torch import Tensor, nn
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram
from transformers import AutoFeatureExtractor, AutoModel
from models.audio.base import AudioEncoder
from models.encoder_types import FrozenEncoderConfig


class DashengAudioEncoder(AudioEncoder):
    def __init__(self, config: FrozenEncoderConfig) -> None:
        super().__init__(config)
        assert config.sample_rate == 16000
        assert config.audio_native_frame_rate == 25 and config.audio_frame_repeat == 2
        directory = str(config.audio_model_directory)
        self.processor = AutoFeatureExtractor.from_pretrained(directory, trust_remote_code=True, local_files_only=True)
        self.encoder = AutoModel.from_pretrained(
            directory, trust_remote_code=True, local_files_only=True, use_safetensors=True, outputdim=None
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
        )
        self.decibels = AmplitudeToDB(top_db=120)

    def features(self, mono: Float[Tensor, "mono sample"]) -> Float[Tensor, "mono feature time"]:
        config = self.config
        assert mono.shape[-1] * config.audio_native_frame_rate % config.sample_rate == 0
        spectrogram = rearrange(
            self.decibels(rearrange(self.spectrogram(mono), "mono mel time -> mono 1 mel time")),
            "mono 1 mel time -> mono mel time",
        )
        assert spectrogram.shape[-1] <= self.encoder.encoder.target_length
        features = self.encoder(input_values=spectrogram).hidden_states
        assert features.shape[1] == mono.shape[-1] * config.audio_native_frame_rate // config.sample_rate
        return repeat(features, "mono native feature -> mono feature (native repeat)", repeat=config.audio_frame_repeat)
