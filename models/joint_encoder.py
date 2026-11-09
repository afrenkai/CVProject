import hashlib
from pathlib import Path
import torch
from beartype import beartype
from einops import rearrange
from huggingface_hub import HfApi, constants, snapshot_download
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn
from torchaudio.functional import resample
from transformers import AutoConfig
from models.audio.audiomosaic import AudioMosaicEncoder
from models.audio.dasheng import DashengAudioEncoder
from models.audio.pe_audio import PeAvAudioEncoder
from models.audio.spatial import BinauralFeatures
from models.encoder_types import AudioMosaicArchitecture, EncoderConfig, FrozenEncoderConfig
from models.video.encoder import DinoVideoEncoder
from models.video.pe_video import PeAvVideoEncoder


class JointEncoder(nn.Module):
    def __init__(self, config: FrozenEncoderConfig, device: torch.device) -> None:
        super().__init__()
        assert config.context_frames % config.video_frames_per_step == 0
        self.settings = config
        self.video_encoder: DinoVideoEncoder
        match config.video_encoder:
            case "dino":
                self.video_encoder = DinoVideoEncoder(
                    config.video_model,
                    True,
                    config.image_size,
                    config.video_layers,
                    (0.485, 0.456, 0.406),
                    (0.229, 0.224, 0.225),
                    frozen=True,
                    cache_directory=config.video_cache_directory,
                )
            case "pe_av":
                video_directory = model_directory(config.video_model, weights=True)
                assert video_directory == config.video_model_directory, f"{config.video_model} -> {video_directory}"
                self.video_encoder = PeAvVideoEncoder(video_directory, config.image_size, config.video_layers)
        self.video_encoder = self.video_encoder.to(device=device, dtype=torch.float32)
        assert self.video_encoder.backbone.embed_dim == config.video_feature_dimension, (
            f"{config.video_model} has width {self.video_encoder.backbone.embed_dim}, "
            f"config video_feature_dimension is {config.video_feature_dimension}"
        )
        self.audio_encoder: DashengAudioEncoder | AudioMosaicEncoder | PeAvAudioEncoder
        match config.audio_encoder:
            case "dasheng":
                self.audio_encoder = DashengAudioEncoder(config, device)
            case "audiomosaic":
                self.audio_encoder = AudioMosaicEncoder(config, device)
            case "pe_av":
                self.audio_encoder = PeAvAudioEncoder(config, device)
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

    def audio_channels(self, waveform: Tensor) -> Tensor:
        config = self.settings
        history = config.audio_history_samples
        window = waveform[..., history:]
        assert window.shape[-1] % config.step_samples == 0
        context_tokens = config.context_frames // config.video_frames_per_step * config.audio_tokens_per_step
        target_tokens = window.shape[-1] // config.step_samples * config.audio_tokens_per_step - context_tokens
        observed = self.audio_encoder(self.semantic_waveform(waveform[:, :2, : history + config.context_samples]))
        complete = self.audio_encoder(self.semantic_waveform(waveform[:, :2])).float()
        semantic = torch.cat(
            (
                observed.float()[..., observed.shape[-1] - context_tokens :],
                complete[..., complete.shape[-1] - target_tokens :],
            ),
            dim=-1,
        )
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
                (window.float(), window.new_zeros(window.shape[0], missing, window.shape[-1]).float()),
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
        config = self.settings
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
        config = self.settings
        assert not self.training and not self.video_encoder.training
        with torch.autocast(frames.device.type, dtype=torch.bfloat16, enabled=config.bfloat16):
            video = rearrange(
                self.video_encoder(frames.float()).float(),
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
        Float[Tensor, "batch time audio_feature"],
    ]:
        video, audio = self.encode_video(frames), self.encode_audio(waveform)
        assert video.shape[:2] == audio.shape[:2]
        return video, audio


def model_directory(name: str, weights: bool = False) -> Path:
    local = Path(name).expanduser()
    if local.is_dir():
        return local
    patterns = ["*.json", "*.py", "*.safetensors"] if weights else ["*.json", "*.py"]
    return Path(snapshot_download(name, allow_patterns=patterns))


def weights_sha256(name: str, directory: Path) -> str:
    """SHA-256 of model.safetensors, identical for a Hub repo and any local copy of it."""
    if not Path(name).expanduser().is_dir() and not constants.HF_HUB_OFFLINE:
        info = HfApi().get_paths_info(name, ["model.safetensors"], revision=directory.name)[0]
        assert info.lfs is not None, f"{name} model.safetensors is not stored with LFS"
        return info.lfs.sha256
    weights = directory / "model.safetensors"
    assert weights.is_file(), f"{weights} is missing; download the weights to identify the checkpoint offline"
    digest = hashlib.sha256()
    with weights.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


#TODO: write custom thing for audio encoder instead
def build_encoder_config(config: EncoderConfig, bfloat16: bool) -> FrozenEncoderConfig:
    audio_path = model_directory(config.audio_model, weights=True)
    sample_rate = 16000
    match config.audio_encoder:
        case "dasheng":
            audio_feature_dimension: int = AutoConfig.from_pretrained(
                str(audio_path), trust_remote_code=True, local_files_only=True
            ).encoder_kwargs["embed_dim"]
            audio_native_frame_rate, audio_frame_repeat, audio_tokens_per_step = 25, 2, 5
            audio_history_steps = 0
            assert not config.audio_frequency_tokens
            audio_token_parts = 1
        case "audiomosaic":
            architecture = AudioMosaicArchitecture.model_validate_json((audio_path / "config.json").read_text())
            audio_feature_dimension = architecture.embed_dim * architecture.spec_size[1] // architecture.patch_size[1]
            assert config.video_rate % config.video_frames_per_step == 0
            audio_native_frame_rate = config.video_rate // config.video_frames_per_step
            audio_frame_repeat, audio_tokens_per_step = 1, 1
            capacity_steps = (
                architecture.capacity_samples * audio_native_frame_rate // architecture.audio_preprocessing.sample_rate
            )
            audio_history_steps = capacity_steps - config.context_frames // config.video_frames_per_step
            frequency_patches = architecture.spec_size[1] // architecture.patch_size[1]
            audio_token_parts = frequency_patches if config.audio_frequency_tokens else 1
        case "pe_av":
            audio_feature_dimension = PeAvAudioEncoder.feature_dimension(audio_path)
            sample_rate = PeAvAudioEncoder.sample_rate
            audio_native_frame_rate, audio_frame_repeat, audio_tokens_per_step = 25, 2, 5
            audio_history_steps = 0
            assert not config.audio_frequency_tokens
            audio_token_parts = 1
    video_path = None if config.video_encoder == "dino" else model_directory(config.video_model)
    return FrozenEncoderConfig(
        video_encoder=config.video_encoder,
        video_model=config.video_model,
        video_model_directory=video_path,
        video_weights=None if video_path is None else weights_sha256(config.video_model, video_path),
        video_feature_dimension=config.video_feature_dimension,
        video_layers=config.video_layers,
        image_size=config.image_size,
        video_rate=config.video_rate,
        video_frames_per_step=config.video_frames_per_step,
        sample_rate=sample_rate,
        context_frames=config.context_frames,
        audio_encoder=config.audio_encoder,
        audio_model_directory=audio_path,
        audio_weights=weights_sha256(config.audio_model, audio_path),
        audio_feature_dimension=audio_feature_dimension,
        audio_native_frame_rate=audio_native_frame_rate,
        audio_frame_repeat=audio_frame_repeat,
        audio_tokens_per_step=audio_tokens_per_step,
        audio_history_steps=audio_history_steps,
        audio_token_parts=audio_token_parts,
        bfloat16=bfloat16,
        spatial_sample_rate=48000 if config.spatial_audio else None,
        ambisonic=config.spatial_audio,
    )
