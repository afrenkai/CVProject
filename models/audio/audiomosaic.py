from functools import partial
import torch
from beartype import beartype
from einops import rearrange
from jaxtyping import Float, jaxtyped
from safetensors.torch import load_file
from timm.layers import PatchEmbed
from timm.models.vision_transformer import VisionTransformer
from torch import Tensor, nn
from torch.nn import functional
from torchaudio.compliance.kaldi import fbank
from models.encoder_types import AudioMosaicArchitecture, FrozenEncoderConfig


class AudioMosaicEncoder(nn.Module):
    def __init__(self, config: FrozenEncoderConfig, device: torch.device) -> None:
        super().__init__()
        architecture = AudioMosaicArchitecture.model_validate_json(
            (config.audio_model_directory / "config.json").read_text()
        )
        preprocessing = architecture.audio_preprocessing
        frames, mels = architecture.spec_size
        assert config.sample_rate == preprocessing.sample_rate and preprocessing.target_length == frames
        frequency_patches = mels // architecture.patch_size[1]
        assert preprocessing.num_mel_bins == mels
        assert architecture.embed_dim * frequency_patches == config.audio_feature_dimension
        assert config.audio_frame_repeat == 1 and config.audio_tokens_per_step == 1
        assert config.sample_rate % config.audio_native_frame_rate == 0
        self.settings = config
        self.architecture = architecture
        layer_norm = partial(nn.LayerNorm, eps=1e-6)
        self.encoder = VisionTransformer(
            img_size=architecture.spec_size,
            patch_size=architecture.patch_size,
            in_chans=1,
            num_classes=0,
            global_pool="",
            embed_dim=architecture.embed_dim,
            depth=architecture.depth,
            num_heads=architecture.num_heads,
            mlp_ratio=architecture.mlp_ratio,
            qkv_bias=True,
            pre_norm=True,
            fc_norm=False,
            norm_layer=layer_norm,
        )
        self.encoder.patch_embed = PatchEmbed(
            img_size=architecture.spec_size,
            patch_size=architecture.patch_size,
            in_chans=1,
            embed_dim=architecture.embed_dim,
            norm_layer=layer_norm,
            bias=True,
        )
        self.encoder.norm = nn.Identity()
        self.output_norm = layer_norm(architecture.embed_dim)
        weights = load_file(config.audio_model_directory / "model.safetensors")
        self.encoder.load_state_dict(
            {name: tensor for name, tensor in weights.items() if not name.startswith(("projector.", "fc_norm."))}
        )
        self.output_norm.load_state_dict({"weight": weights["fc_norm.weight"], "bias": weights["fc_norm.bias"]})
        self.to(device=device, dtype=torch.float32).eval().requires_grad_(False)

    @torch.no_grad()
    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        waveform: Float[Tensor, "batch channel sample"],
    ) -> Float[Tensor, "mono_batch feature time"]:
        config = self.settings
        preprocessing = self.architecture.audio_preprocessing
        frames, mels = self.architecture.spec_size
        patch_frames = self.architecture.patch_size[0]
        time_patches = frames // patch_frames
        architecture = self.architecture
        step_samples = config.sample_rate // config.audio_native_frame_rate
        assert waveform.shape[1] == 2 and torch.isfinite(waveform).all()
        assert waveform.shape[-1] % step_samples == 0
        kept = min(waveform.shape[-1], architecture.capacity_samples // step_samples * step_samples)
        mono = rearrange(waveform[..., -kept:].float(), "batch channel sample -> (batch channel) sample")
        padding = architecture.capacity_samples - mono.shape[-1]
        padded = functional.pad(mono - mono.mean(dim=-1, keepdim=True), (padding, 0))
        with torch.autocast(waveform.device.type, enabled=False):
            spectrogram = torch.stack(
                [
                    fbank(
                        rearrange(clip, "sample -> 1 sample"),
                        htk_compat=True,
                        sample_frequency=config.sample_rate,
                        use_energy=False,
                        window_type="hanning",
                        num_mel_bins=mels,
                        dither=0.0,
                        frame_shift=preprocessing.frame_shift,
                    )
                    for clip in padded
                ]
            )
        assert spectrogram.shape[1:] == (frames, mels)
        normalized = (spectrogram - preprocessing.norm_mean) / preprocessing.norm_std
        tokens = self.encoder.forward_features(rearrange(normalized, "mono frame mel -> mono 1 frame mel"))
        series = rearrange(
            self.output_norm(tokens[:, self.encoder.num_prefix_tokens :].float()),
            "mono (time frequency) feature -> mono time (frequency feature)",
            time=time_patches,
        )
        steps = mono.shape[-1] // step_samples
        step_centers = padding + step_samples * torch.arange(steps, device=waveform.device) + step_samples / 2
        patch_center = (architecture.window_samples + (patch_frames - 1) * architecture.frame_shift_samples) / 2
        position = ((step_centers - patch_center) / (patch_frames * architecture.frame_shift_samples)).clamp(
            0, time_patches - 1
        )
        lower = position.floor().long()
        upper = (lower + 1).clamp(max=time_patches - 1)
        resampled = torch.lerp(
            series[:, lower],
            series[:, upper],
            rearrange(position - lower, "time -> 1 time 1"),
        )
        assert resampled.shape == (mono.shape[0], steps, config.audio_feature_dimension)
        assert torch.isfinite(resampled).all()
        return rearrange(resampled, "mono time feature -> mono feature time")
