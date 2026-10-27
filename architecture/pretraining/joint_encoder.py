import torch
from beartype import beartype
from einops import rearrange
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn
from architecture.pretraining.audio.dasheng import DashengAudioEncoder
from architecture.pretraining.encoder_types import FrozenEncoderConfig
from architecture.pretraining.video.encoder import DinoVideoEncoder


class JointEncoder(nn.Module):
    def __init__(self, config: FrozenEncoderConfig, device: torch.device) -> None:
        super().__init__()
        assert config.context_frames % config.video_frames_per_step == 0
        self.settings = config
        self.video_encoder = DinoVideoEncoder(
            config.video_model,
            True,
            config.image_size,
            config.video_layers,
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
            frozen=True,
            cache_directory=config.video_cache_directory,
        ).to(device=device, dtype=torch.float32)
        self.audio_encoder = DashengAudioEncoder(config, device)
        self.eval().requires_grad_(False)
        assert all(parameter.dtype == torch.float32 and not parameter.requires_grad for parameter in self.parameters())

    @torch.no_grad()
    @jaxtyped(typechecker=beartype)
    def encode_observed(
        self,
        frames: Float[Tensor, "batch frame video_channel height width"],
        waveform: Float[Tensor, "batch audio_channel sample"],
    ) -> tuple[
        Float[Tensor, "batch time patch video_feature"],
        Float[Tensor, "batch time audio_feature"],
    ]:
        config = self.settings
        assert not self.training and not self.video_encoder.training and not self.audio_encoder.training
        context_samples = config.context_frames * config.sample_rate // config.video_rate
        assert config.context_frames * config.sample_rate % config.video_rate == 0
        assert frames.shape[1] >= config.context_frames and waveform.shape[-1] >= context_samples
        with torch.autocast(frames.device.type, enabled=False):
            observed_video = self.video_encoder(frames[:, : config.context_frames].float())
            observed_audio = self.audio_encoder(waveform[..., :context_samples].float())
            video = rearrange(
                observed_video,
                "batch (time phase) patch feature -> batch time patch (phase feature)",
                phase=config.video_frames_per_step,
            )
            audio = rearrange(
                observed_audio,
                "(batch channel) feature (time phase) -> batch time (channel phase feature)",
                channel=2,
                phase=config.audio_tokens_per_step,
            )
        assert video.shape[1] == audio.shape[1] == config.context_frames // config.video_frames_per_step
        assert video.dtype == audio.dtype == torch.float32
        assert torch.isfinite(video).all() and torch.isfinite(audio).all()
        return video, audio

    @torch.no_grad()
    @jaxtyped(typechecker=beartype)
    def encode(
        self,
        frames: Float[Tensor, "batch frame video_channel height width"],
        waveform: Float[Tensor, "batch audio_channel sample"],
    ) -> tuple[
        Float[Tensor, "batch time patch video_feature"],
        Float[Tensor, "batch time audio_feature"],
    ]:
        config = self.settings
        assert not self.training and not self.video_encoder.training and not self.audio_encoder.training
        context_steps = config.context_frames // config.video_frames_per_step
        context_tokens = context_steps * config.audio_tokens_per_step
        observed_video, observed_audio = self.encode_observed(frames, waveform)
        with torch.autocast(frames.device.type, enabled=False):
            complete_video = rearrange(
                self.video_encoder(frames.float()),
                "batch (time phase) patch feature -> batch time patch (phase feature)",
                phase=config.video_frames_per_step,
            )
            complete_audio = self.audio_encoder(waveform.float())
            future_audio = rearrange(
                complete_audio[..., context_tokens:],
                "(batch channel) feature (time phase) -> batch time (channel phase feature)",
                channel=2,
                phase=config.audio_tokens_per_step,
            )
            video = torch.cat((observed_video, complete_video[:, context_steps:]), dim=1)
            audio = torch.cat((observed_audio, future_audio), dim=1)
        assert video.shape[:2] == audio.shape[:2]
        assert video.dtype == audio.dtype == torch.float32
        assert torch.isfinite(video).all() and torch.isfinite(audio).all()
        return video, audio
