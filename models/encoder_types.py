from pathlib import Path
from typing import Literal
from pydantic import BaseModel, ConfigDict, NonNegativeInt, PositiveFloat, PositiveInt
from configs.config_types import FrozenModel

AudioEncoderName = Literal["dasheng", "audiomosaic"]


class AudioMosaicPreprocessing(FrozenModel):
    target_length: PositiveInt
    num_mel_bins: PositiveInt
    norm_mean: float
    norm_std: PositiveFloat
    sample_rate: PositiveInt
    frame_shift: PositiveFloat


class AudioMosaicArchitecture(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    spec_size: tuple[PositiveInt, PositiveInt]
    patch_size: tuple[PositiveInt, PositiveInt]
    embed_dim: PositiveInt
    depth: PositiveInt
    num_heads: PositiveInt
    mlp_ratio: PositiveFloat
    audio_preprocessing: AudioMosaicPreprocessing

    @property
    def frame_shift_samples(self) -> int:
        return round(self.audio_preprocessing.sample_rate * self.audio_preprocessing.frame_shift / 1000)

    @property
    def window_samples(self) -> int:
        return self.audio_preprocessing.sample_rate * 25 // 1000

    @property
    def capacity_samples(self) -> int:
        return self.window_samples + (self.spec_size[0] - 1) * self.frame_shift_samples


class EncoderConfig(FrozenModel):
    video_model: str = "vit_large_patch16_dinov3.lvd1689m"
    video_feature_dimension: PositiveInt = 1024
    video_layers: tuple[PositiveInt, ...] = (6, 12, 18, 24)
    image_size: PositiveInt = 256
    audio_encoder: AudioEncoderName = "dasheng"
    audio_model: str = "mispeech/dasheng-1.2B"
    audio_frequency_tokens: bool = False
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
    audio_encoder: AudioEncoderName = "dasheng"
    audio_model_directory: Path
    audio_feature_dimension: PositiveInt = 1536
    audio_native_frame_rate: PositiveInt = 25
    audio_frame_repeat: PositiveInt = 2
    audio_tokens_per_step: PositiveInt = 5
    audio_history_steps: NonNegativeInt = 0
    audio_token_parts: PositiveInt = 1
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
    def step_samples(self) -> int:
        return self.video_frames_per_step * self.loading_sample_rate // self.video_rate

    @property
    def context_samples(self) -> int:
        return self.context_frames * self.loading_sample_rate // self.video_rate

    @property
    def audio_history_samples(self) -> int:
        return self.audio_history_steps * self.step_samples

    @property
    def video_latent_dimension(self) -> int:
        return self.video_feature_dimension * self.video_frames_per_step

    @property
    def audio_latent_dimension(self) -> int:
        return self.audio_feature_dimension * self.audio_channels * self.audio_tokens_per_step
