from pathlib import Path
from typing import Annotated, Literal
from huggingface_hub import snapshot_download
from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, PositiveFloat, PositiveInt
from transformers import AutoConfig
from configs.config_types import FrozenModel


class StepTiming(FrozenModel):
    video_rate: PositiveInt
    video_frames_per_step: PositiveInt
    context_frames: PositiveInt


class AudioTiming(FrozenModel):
    audio_feature_dimension: PositiveInt
    audio_native_frame_rate: PositiveInt
    audio_frame_repeat: PositiveInt
    audio_tokens_per_step: PositiveInt
    audio_history_steps: NonNegativeInt
    audio_token_parts: PositiveInt
    sample_rate: PositiveInt = 16000


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
        # TODO: make adaptive
        return self.audio_preprocessing.sample_rate * 25 // 1000

    @property
    def capacity_samples(self) -> int:
        return self.window_samples + (self.spec_size[0] - 1) * self.frame_shift_samples


class Dinov3Config(FrozenModel):
    kind: Literal["dinov3"] = "dinov3"
    model_id: str = "vit_large_patch16_dinov3.lvd1689m"
    feature_dimension: PositiveInt = 1024
    layers: tuple[PositiveInt, ...] = (6, 12, 18, 24)
    patch_size: PositiveInt = 16
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    standard_deviation: tuple[float, float, float] = (0.229, 0.224, 0.225)
    cache_directory: Path | None = None


class PeAvVideoConfig(FrozenModel):
    kind: Literal["pe_av"] = "pe_av"
    model_id: str = "facebook/pe-av-base"
    feature_dimension: PositiveInt = 1024
    layers: tuple[PositiveInt, ...] = (6, 12, 18, 24)
    patch_size: PositiveInt = 16


class DashengConfig(FrozenModel):
    kind: Literal["dasheng"] = "dasheng"
    model_id: str = "mispeech/dasheng-1.2B"

    def timing(self, directory: Path, steps: StepTiming) -> AudioTiming:
        pretrained = AutoConfig.from_pretrained(str(directory), trust_remote_code=True, local_files_only=True)
        return AudioTiming(
            audio_feature_dimension=pretrained.encoder_kwargs["embed_dim"],
            audio_native_frame_rate=25,
            audio_frame_repeat=2,
            audio_tokens_per_step=5,
            audio_history_steps=0,
            audio_token_parts=1,
        )


class AudioMosaicConfig(FrozenModel):
    kind: Literal["audiomosaic"] = "audiomosaic"
    model_id: str = "hanxunh/AudioMosaic-vit-b16-pretrained"
    frequency_tokens: bool = False

    def timing(self, directory: Path, steps: StepTiming) -> AudioTiming:
        # checking via hf
        architecture = AudioMosaicArchitecture.model_validate_json((directory / "config.json").read_text())

        assert steps.video_rate % steps.video_frames_per_step == 0
        native_frame_rate = steps.video_rate // steps.video_frames_per_step
        capacity_steps = (
            architecture.capacity_samples * native_frame_rate // architecture.audio_preprocessing.sample_rate
        )
        frequency_patches = architecture.spec_size[1] // architecture.patch_size[1]
        return AudioTiming(
            audio_feature_dimension=architecture.embed_dim * frequency_patches,
            audio_native_frame_rate=native_frame_rate,
            audio_frame_repeat=1,
            audio_tokens_per_step=1,
            audio_history_steps=capacity_steps - steps.context_frames // steps.video_frames_per_step,
            audio_token_parts=frequency_patches if self.frequency_tokens else 1,
        )


class PeAvAudioConfig(FrozenModel):
    kind: Literal["pe_av"] = "pe_av"
    model_id: str = "facebook/pe-av-base"

    def timing(self, directory: Path, steps: StepTiming) -> AudioTiming:
        audio = AutoConfig.from_pretrained(str(directory), local_files_only=True).audio_video_config.audio_config
        return AudioTiming(
            audio_feature_dimension=audio.hidden_size,
            audio_native_frame_rate=audio.dac_config.sampling_rate // audio.dac_config.hop_length,
            audio_frame_repeat=2,
            audio_tokens_per_step=5,
            audio_history_steps=0,
            audio_token_parts=1,
            sample_rate=audio.dac_config.sampling_rate,
        )


type VideoEncoderSettings = Annotated[Dinov3Config | PeAvVideoConfig, Field(discriminator="kind")]
type AudioEncoderSettings = Annotated[DashengConfig | AudioMosaicConfig | PeAvAudioConfig, Field(discriminator="kind")]


class FrozenEncoderConfig(AudioTiming):
    video: VideoEncoderSettings
    audio: AudioEncoderSettings
    audio_model_directory: Path
    image_size: PositiveInt = 128
    video_rate: PositiveInt = 30
    video_frames_per_step: PositiveInt = 3
    context_frames: PositiveInt = 60
    bfloat16: bool = False
    spatial_sample_rate: PositiveInt | None = None
    spatial_mel_bands: PositiveInt = 64
    maximum_interaural_delay_seconds: float = 0.001
    ambisonic: bool = False
    kept_audio_history_steps: NonNegativeInt = 0
    audio_target_chunk_steps: PositiveInt | None = None
    audio_window_steps: PositiveInt | None = None

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
    def video_patches(self) -> int:
        return (self.image_size // self.video.patch_size) ** 2

    @property
    def video_latent_dimension(self) -> int:
        return self.video.feature_dimension * self.video_frames_per_step

    @property
    def audio_latent_dimension(self) -> int:
        return self.audio_feature_dimension * self.audio_channels * self.audio_tokens_per_step


class EncoderConfig(FrozenModel):
    video: VideoEncoderSettings = Field(default_factory=Dinov3Config)
    audio: AudioEncoderSettings = Field(default_factory=DashengConfig)
    image_size: PositiveInt = 256
    video_rate: PositiveInt = 10
    video_frames_per_step: PositiveInt = 1
    context_frames: PositiveInt = 12
    total_frames: PositiveInt = 40
    spatial_audio: bool = False
    audio_history_frames: NonNegativeInt = 0
    audio_target_chunk_frames: PositiveInt | None = None
    audio_window_frames: PositiveInt | None = None

    def resolve(self, bfloat16: bool) -> FrozenEncoderConfig:
        directory = Path(snapshot_download(self.audio.model_id))
        steps = StepTiming(
            video_rate=self.video_rate,
            video_frames_per_step=self.video_frames_per_step,
            context_frames=self.context_frames,
        )
        timing = self.audio.timing(directory, steps)
        assert self.audio_history_frames % self.video_frames_per_step == 0
        kept = self.audio_history_frames // self.video_frames_per_step
        assert kept == 0 or timing.audio_history_steps == 0
        chunk = self.audio_target_chunk_frames
        assert chunk is None or chunk % self.video_frames_per_step == 0
        window = self.audio_window_frames
        assert window is None or window % self.video_frames_per_step == 0
        return FrozenEncoderConfig(
            **timing.model_copy(update={"audio_history_steps": timing.audio_history_steps + kept}).model_dump(),
            kept_audio_history_steps=kept,
            audio_target_chunk_steps=None if chunk is None else chunk // self.video_frames_per_step,
            audio_window_steps=None if window is None else window // self.video_frames_per_step,
            video=self.video,
            audio=self.audio,
            audio_model_directory=directory,
            image_size=self.image_size,
            video_rate=self.video_rate,
            video_frames_per_step=self.video_frames_per_step,
            context_frames=self.context_frames,
            bfloat16=bfloat16,
            spatial_sample_rate=48000 if self.spatial_audio else None,
            ambisonic=self.spatial_audio,
        )
