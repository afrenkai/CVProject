import torch
from beartype import beartype
from einops import rearrange
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn
from torchaudio.functional import resample
from models.audio.audiomosaic import AudioMosaicEncoder
from models.audio.base import AudioEncoder
from models.audio.dasheng import DashengAudioEncoder
from models.audio.pe_audio import PeAvAudioEncoder
from models.audio.spatial import BinauralFeatures
from models.encoder_types import (
    AudioMosaicConfig,
    DashengConfig,
    Dinov3Config,
    FrozenEncoderConfig,
    PeAvAudioConfig,
    PeAvVideoConfig,
)
from models.video.base import VideoEncoder
from models.video.dinov3 import Dinov3VideoEncoder
from models.video.pe_video import PeAvVideoEncoder


def build_audio_encoder(config: FrozenEncoderConfig, device: torch.device) -> AudioEncoder:
    match config.audio:
        case DashengConfig():
            return DashengAudioEncoder(config).freeze(device)
        case AudioMosaicConfig():
            return AudioMosaicEncoder(config).freeze(device)
        case PeAvAudioConfig():
            return PeAvAudioEncoder(config).freeze(device)


def build_video_encoder(config: FrozenEncoderConfig, device: torch.device) -> VideoEncoder:
    match config.video:
        case Dinov3Config() as video:
            return Dinov3VideoEncoder(config, video).freeze(device)
        case PeAvVideoConfig() as video:
            return PeAvVideoEncoder(config, video).freeze(device)


class JointEncoder(nn.Module):
    def __init__(self, config: FrozenEncoderConfig, device: torch.device) -> None:
        super().__init__()
        assert config.context_frames % config.video_frames_per_step == 0
        self.config = config
        self.video_encoder = build_video_encoder(config, device)
        self.audio_encoder = build_audio_encoder(config, device)
        self.spatial_features = (
            None
            if config.spatial_sample_rate is None
            else BinauralFeatures(
                config.spatial_sample_rate,
                config.audio_native_frame_rate * config.audio_frame_repeat,
                config.spatial_mel_bands,
                config.maximum_interaural_delay_seconds,
                config.audio_feature_dimension,
                config.ambisonic,
            ).to(device)
        )
        self.eval().requires_grad_(False)
        # surely nothing goes wrong here
        assert all(parameter.dtype == torch.float32 and not parameter.requires_grad for parameter in self.parameters())

    def semantic_waveform(self, waveform: Tensor) -> Tensor:
        config = self.config
        if config.spatial_sample_rate is None:
            return waveform.float().contiguous()
        with torch.autocast(waveform.device.type, enabled=False):
            return resample(waveform.float().contiguous(), config.spatial_sample_rate, config.sample_rate)

    def audio_window(self, waveform: Tensor, end: int) -> Tensor:
        window = self.config.audio_window_steps
        begin = 0 if window is None else max(0, end - window * self.config.step_samples)
        return self.audio_encoder(self.semantic_waveform(waveform[:, :2, begin:end])).float()

    def audio_channels(self, waveform: Tensor) -> Tensor:
        config = self.config
        history = config.audio_history_samples
        clip = waveform[..., history:]
        assert clip.shape[-1] % config.step_samples == 0
        tokens = config.audio_tokens_per_step
        context_end = history + config.context_samples
        kept_tokens = (config.kept_audio_history_steps + config.context_frames // config.video_frames_per_step) * tokens
        target_steps = clip.shape[-1] // config.step_samples - config.context_frames // config.video_frames_per_step
        chunk = config.audio_target_chunk_steps or target_steps
        observed = self.audio_window(waveform, context_end)
        parts = [observed[..., observed.shape[-1] - kept_tokens :]]
        for chunk_start in range(0, target_steps, chunk):
            chunk_steps = min(chunk, target_steps - chunk_start)
            encoded = self.audio_window(waveform, context_end + (chunk_start + chunk_steps) * config.step_samples)
            parts.append(encoded[..., encoded.shape[-1] - chunk_steps * tokens :])
        semantic = torch.cat(parts, dim=-1)
        channels = rearrange(
            semantic,
            "(batch channel) feature time -> batch channel feature time",
            channel=2,
        )
        if self.spatial_features is None:
            return channels
        missing = config.waveform_channels - waveform.shape[1]
        spatial_input = waveform[..., history - config.kept_audio_history_steps * config.step_samples :]
        spatial = self.spatial_features(
            torch.cat(
                (
                    spatial_input.float(),
                    spatial_input.new_zeros(spatial_input.shape[0], missing, spatial_input.shape[-1]).float(),
                ),
                dim=1,
            )
        )
        assert spatial.shape[-1] >= channels.shape[-1]
        return torch.cat(
            (
                channels,
                rearrange(spatial[..., : channels.shape[-1]], "batch f t -> batch 1 f t"),
            ),
            dim=1,
        )

    @torch.no_grad()
    @jaxtyped(typechecker=beartype)
    def encode_audio(
        self,
        waveform: Float[Tensor, "batch audio_channel sample"],
    ) -> Float[Tensor, "batch time audio_feature"]:
        config = self.config
        assert not self.training and not self.audio_encoder.training
        with torch.autocast(waveform.device.type, dtype=torch.bfloat16, enabled=config.bfloat16):
            audio = rearrange(
                self.audio_channels(waveform),
                "batch channel feature (time phase) -> batch time (channel phase feature)",
                phase=config.audio_tokens_per_step,
            )
        assert audio.dtype == torch.float32 and torch.isfinite(audio).all()
        return audio

    @torch.no_grad()
    @jaxtyped(typechecker=beartype)
    def encode_video(
        self,
        frames: Float[Tensor, "batch frame video_channel height width"],
    ) -> Float[Tensor, "batch time patch video_feature"]:
        config = self.config
        assert not self.training and not self.video_encoder.training
        with torch.autocast(frames.device.type, dtype=torch.bfloat16, enabled=config.bfloat16):
            video = rearrange(
                self.video_encoder(frames).float(),
                "batch (time phase) patch feature -> batch time patch (phase feature)",
                phase=config.video_frames_per_step,
            )
        assert video.dtype == torch.float32 and torch.isfinite(video).all()
        return video

    @torch.no_grad()
    @jaxtyped(typechecker=beartype)
    def encode(
        self,
        frames: Float[Tensor, "batch frame video_channel height width"],
        waveform: Float[Tensor, "batch audio_channel sample"],
    ) -> tuple[
        Float[Tensor, "batch time patch video_feature"],
        Float[Tensor, "batch audio_time audio_feature"],
    ]:
        video, audio = self.encode_video(frames), self.encode_audio(waveform)
        assert video.shape[0] == audio.shape[0]
        assert audio.shape[1] == video.shape[1] + self.config.kept_audio_history_steps
        return video, audio
