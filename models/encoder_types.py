from pathlib import Path
from pydantic import PositiveInt
from configs.config_types import FrozenModel


class EncoderConfig(FrozenModel):
    video_model: str = "vit_large_patch16_dinov3.lvd1689m"
    video_feature_dimension: PositiveInt = 1024
    video_layers: tuple[PositiveInt, ...] = (6, 12, 18, 24)
    image_size: PositiveInt = 256
    audio_model: str = "mispeech/dasheng-1.2B"
    video_rate: PositiveInt = 10
    video_frames_per_step: PositiveInt = 1
    context_frames: PositiveInt = 12
    total_frames: PositiveInt = 40
    spatial_audio: bool = False


class FrozenEncoderConfig(FrozenModel):
    video_model: str = "vit_large_patch16_dinov3.lvd1689m"
    video_cache_directory: Path | None = None
    video_layers: tuple[PositiveInt, ...] = (6, 12, 18, 24)
    video_feature_dimension: PositiveInt = 1024
    image_size: PositiveInt = 128
    video_rate: PositiveInt = 30
    sample_rate: PositiveInt = 16000
    video_frames_per_step: PositiveInt = 3
    context_frames: PositiveInt = 60
    audio_model_directory: Path
    audio_feature_dimension: PositiveInt = 1536
    audio_native_frame_rate: PositiveInt = 25
    audio_frame_repeat: PositiveInt = 2
    audio_tokens_per_step: PositiveInt = 5
    bfloat16: bool = False
    spatial_sample_rate: PositiveInt | None = None
    spatial_mel_bands: PositiveInt = 64
    maximum_interaural_delay_seconds: float = 0.001
    ambisonic: bool = False

    @property
    def audio_channels(self) -> int:
        return 2 if self.spatial_sample_rate is None else 3

    @property
    def waveform_channels(self) -> int:
        return 4 if self.ambisonic else 2

    @property
    def loading_sample_rate(self) -> int:
        return self.spatial_sample_rate or self.sample_rate

    @property
    def video_latent_dimension(self) -> int:
        return self.video_feature_dimension * self.video_frames_per_step

    @property
    def audio_latent_dimension(self) -> int:
        return self.audio_feature_dimension * self.audio_channels * self.audio_tokens_per_step
