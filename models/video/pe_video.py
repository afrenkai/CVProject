from collections.abc import Sequence
from pathlib import Path
import timm
from safetensors import safe_open
from transformers import AutoConfig
from models.video.encoder import DinoVideoEncoder


class PeAvVideoEncoder(DinoVideoEncoder):
    weight_prefix = "video_model.video_encoder.embedder.vision_model.timm_model."

    def __init__(self, model_directory: Path, image_size: int, layer_numbers: Sequence[int]) -> None:
        """
        PE-AV's per-frame PE-Core ViT as a drop-in for DINO patch features
        weights load at the native resolution, then the input is set to (image_size // 16) * patch
        so the grid matches the (image_size // 16) ** 2 patches the rest of the model expects
        """
        vision_config = AutoConfig.from_pretrained(
            str(model_directory), local_files_only=True
        ).audio_video_config.video_config.vision_config
        native = timm.models.get_pretrained_cfg(vision_config.architecture)
        assert native is not None, f"timm has no pretrained config for {vision_config.architecture}"
        super().__init__(
            vision_config.architecture,
            False,
            native.input_size[-1],
            layer_numbers,
            native.mean,
            native.std,
            frozen=True,
        )
        with safe_open(str(model_directory / "model.safetensors"), "pt") as handle:
            weights = {
                name.removeprefix(self.weight_prefix): handle.get_tensor(name)
                for name in handle.keys()  # noqa: SIM118
                if name.startswith(self.weight_prefix) and not name.startswith(self.weight_prefix + "head.")
            }
        self.backbone.load_state_dict(weights, strict=True)
        grid = image_size // 16
        patch_size = self.backbone.patch_embed.patch_size[0]
        self.backbone.set_input_size(img_size=(grid * patch_size, grid * patch_size))
        assert tuple(self.backbone.patch_embed.grid_size) == (grid, grid)
        self.image_size = grid * patch_size
        self.requires_grad_(False)
