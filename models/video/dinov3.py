from typing import cast
import timm
import torch
from einops import reduce
from jaxtyping import Float
from timm.models.vision_transformer import VisionTransformer
from torch import Tensor
from models.encoder_types import Dinov3Config, FrozenEncoderConfig
from models.video.base import VideoEncoder


def layer_mean(
    backbone: VisionTransformer, frames: Float[Tensor, "frame channel height width"], layer_indices: list[int]
) -> Float[Tensor, "frame patch feature"]:
    layers = cast(
        list[Tensor],
        backbone.forward_intermediates(
            frames, indices=layer_indices, norm=True, output_fmt="NLC", intermediates_only=True
        ),
    )
    return reduce(torch.stack(layers), "layer frame patch feature -> frame patch feature", "mean")


class Dinov3VideoEncoder(VideoEncoder):
    def __init__(self, config: FrozenEncoderConfig, video: Dinov3Config) -> None:
        """
        in general, dinoV2/V3 returns mean pooled lat of cls_tok, r_1 .. r_4
        you can check fn_signature and look at a sample output
        norm is imagenet norm usually
        but could also be 3 ch of 0.5 (not really standard)
        """
        super().__init__(config)
        assert min(video.layers) >= 1
        self.backbone = cast(
            VisionTransformer,
            timm.create_model(
                video.model_id,
                pretrained=True,
                num_classes=0,
                img_size=config.image_size,
                cache_dir=video.cache_directory,
            ),
        )
        self.layer_indices = [layer - 1 for layer in video.layers]
        self.register_buffer("mean", torch.tensor(video.mean)[None, :, None, None], persistent=False)
        self.register_buffer(
            "standard_deviation", torch.tensor(video.standard_deviation)[None, :, None, None], persistent=False
        )

    def features(self, frames: Float[Tensor, "frame channel height width"]) -> Float[Tensor, "frame patch feature"]:
        return layer_mean(self.backbone, (frames - self.mean) / self.standard_deviation, self.layer_indices)
