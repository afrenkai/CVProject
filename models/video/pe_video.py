from pathlib import Path
from typing import cast
import timm
import torch
from huggingface_hub import snapshot_download
from jaxtyping import Float
from safetensors import safe_open
from timm.models.vision_transformer import VisionTransformer
from torch import Tensor
from transformers import AutoConfig
from models.encoder_types import FrozenEncoderConfig, PeAvVideoConfig
from models.video.base import VideoEncoder
from models.video.dinov3 import layer_mean


class PeAvVideoEncoder(VideoEncoder):
    weight_prefix = "video_model.video_encoder.embedder.vision_model.timm_model."

    def __init__(self, config: FrozenEncoderConfig, video: PeAvVideoConfig) -> None:
        """
        PE-AV's per-frame PE-Core ViT as a drop-in for DINO patch features
        weights load at the native resolution, then the input is set to (image_size // patch_size) * native patch
        so the grid matches the video_patches the rest of the model expects
        """
        super().__init__(config)
        assert min(video.layers) >= 1
        directory = Path(snapshot_download(video.model_id, allow_patterns=["*.json", "*.safetensors"]))
        architecture = AutoConfig.from_pretrained(
            str(directory), local_files_only=True
        ).audio_video_config.video_config.vision_config.architecture
        native = timm.models.get_pretrained_cfg(architecture)
        assert native is not None, f"timm has no pretrained config for {architecture}"
        self.backbone = cast(
            VisionTransformer,
            timm.create_model(architecture, pretrained=False, num_classes=0, img_size=native.input_size[-1]),
        )
        with safe_open(str(directory / "model.safetensors"), "pt") as handle:
            weights = {
                name.removeprefix(self.weight_prefix): handle.get_tensor(name)
                for name in handle.keys()  # noqa: SIM118
                if name.startswith(self.weight_prefix) and not name.startswith(self.weight_prefix + "head.")
            }
        self.backbone.load_state_dict(weights, strict=True)
        assert self.backbone.embed_dim == video.feature_dimension
        grid = config.image_size // video.patch_size
        native_patch = self.backbone.patch_embed.patch_size[0]
        self.backbone.set_input_size(img_size=(grid * native_patch, grid * native_patch))
        assert tuple(self.backbone.patch_embed.grid_size) == (grid, grid)
        self.input_size = grid * native_patch
        self.layer_indices = [layer - 1 for layer in video.layers]
        self.register_buffer("mean", torch.tensor(native.mean)[None, :, None, None], persistent=False)
        self.register_buffer("standard_deviation", torch.tensor(native.std)[None, :, None, None], persistent=False)

    def features(self, frames: Float[Tensor, "frame channel height width"]) -> Float[Tensor, "frame patch feature"]:
        return layer_mean(self.backbone, (frames - self.mean) / self.standard_deviation, self.layer_indices)
