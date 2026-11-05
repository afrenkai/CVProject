from pathlib import Path
import torch
from beartype import beartype
from einops import rearrange
from huggingface_hub import snapshot_download
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn
from torchaudio.functional import resample
from transformers import AutoConfig
from models.audio.dasheng import DashengAudioEncoder
from models.audio.spatial import BinauralFeatures
from models.encoder_types import EncoderConfig, FrozenEncoderConfig
from models.video.encoder import DinoVideoEncoder


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
        assert all(parameter.dtype == torch.float32 and not parameter.requires_grad for parameter in self.parameters())

    def semantic_waveform(self, waveform: Tensor) -> Tensor:
        config = self.settings
        if config.spatial_sample_rate is None:
            return waveform.float().contiguous()
        with torch.autocast(waveform.device.type, enabled=False):
            return resample(waveform.float().contiguous(), config.spatial_sample_rate, config.sample_rate)

    def audio_channels(self, waveform: Tensor, context_only: bool) -> Tensor:
        config = self.settings
        context_samples = config.context_frames * config.loading_sample_rate // config.video_rate
        context_tokens = config.context_frames // config.video_frames_per_step * config.audio_tokens_per_step
        observed = self.audio_encoder(self.semantic_waveform(waveform[:, :2, :context_samples])).float()
        if context_only:
            semantic = observed
        else:
            complete = self.audio_encoder(self.semantic_waveform(waveform[:, :2])).float()
            semantic = torch.cat((observed, complete[..., context_tokens:]), dim=-1)
        channels = rearrange(
            semantic,
            "(batch channel) feature time -> batch channel feature time",
            channel=2,
        )
        if self.spatial_features is None:
            return channels
        missing = config.waveform_channels - waveform.shape[1]
        spatial = self.spatial_features(
            torch.cat(
                (waveform.float(), waveform.new_zeros(waveform.shape[0], missing, waveform.shape[-1]).float()),
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
        context_samples = config.context_frames * config.loading_sample_rate // config.video_rate
        assert config.context_frames * config.loading_sample_rate % config.video_rate == 0
        assert frames.shape[1] >= config.context_frames and waveform.shape[-1] >= context_samples
        with torch.autocast(frames.device.type, dtype=torch.bfloat16, enabled=config.bfloat16):
            observed_video = self.video_encoder(frames[:, : config.context_frames].float()).float()
            video = rearrange(
                observed_video,
                "batch (time phase) patch feature -> batch time patch (phase feature)",
                phase=config.video_frames_per_step,
            )
            audio = rearrange(
                self.audio_channels(waveform[..., :context_samples], context_only=True),
                "batch channel feature (time phase) -> batch time (channel phase feature)",
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
        with torch.autocast(frames.device.type, dtype=torch.bfloat16, enabled=config.bfloat16):
            video = rearrange(
                self.video_encoder(frames.float()).float(),
                "batch (time phase) patch feature -> batch time patch (phase feature)",
                phase=config.video_frames_per_step,
            )
            audio = rearrange(
                self.audio_channels(waveform, context_only=False),
                "batch channel feature (time phase) -> batch time (channel phase feature)",
                phase=config.audio_tokens_per_step,
            )
        assert video.shape[:2] == audio.shape[:2]
        assert video.dtype == audio.dtype == torch.float32
        assert torch.isfinite(video).all() and torch.isfinite(audio).all()
        return video, audio


def build_encoder_config(config: EncoderConfig, bfloat16: bool) -> FrozenEncoderConfig:
    audio_path = Path(snapshot_download(config.audio_model))
    audio_feature_dimension: int = AutoConfig.from_pretrained(
        str(audio_path), trust_remote_code=True, local_files_only=True
    ).encoder_kwargs["embed_dim"]
    return FrozenEncoderConfig(
        video_model=config.video_model,
        video_feature_dimension=config.video_feature_dimension,
        video_layers=config.video_layers,
        image_size=config.image_size,
        video_rate=config.video_rate,
        video_frames_per_step=config.video_frames_per_step,
        sample_rate=16000,
        context_frames=config.context_frames,
        audio_model_directory=audio_path,
        audio_feature_dimension=audio_feature_dimension,
        audio_native_frame_rate=25,
        audio_frame_repeat=2,
        audio_tokens_per_step=5,
        bfloat16=bfloat16,
        spatial_sample_rate=48000 if config.spatial_audio else None,
        ambisonic=config.spatial_audio,
    )
