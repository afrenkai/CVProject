from collections.abc import Sequence
from beartype import beartype
from einops import rearrange
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn
from torch.nn import functional as F


class DecoderResidualBlock(nn.Module):
    def __init__(self, channels: int, normalization_groups: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(normalization_groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(normalization_groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, features: Tensor) -> Tensor:
        return features + self.block(features)


class DecoderUpsampleBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, normalization_groups: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1),
            nn.GroupNorm(normalization_groups, output_channels),
            nn.SiLU(),
        )

    def forward(self, features: Tensor) -> Tensor:
        features = F.interpolate(
            features,
            scale_factor=2.0,
            mode="bilinear",
            align_corners=False,
        )
        return self.block(features)


class DinoVideoDecoder(nn.Module):
    def __init__(
        self,
        feature_dimension: int,
        hidden_dimension: int,
        upsample_channels: Sequence[int],
        residual_blocks: int,
        normalization_groups: int,
        patch_grid_size: int,
    ) -> None:
        super().__init__()
        channels = (hidden_dimension, *upsample_channels)
        if any(channel % normalization_groups for channel in channels):
            raise ValueError("Decoder channels must divide evenly into normalization groups")
        if residual_blocks <= 0 or patch_grid_size <= 0:
            raise ValueError("Decoder block count and patch grid must be positive")
        self.patch_grid_size = patch_grid_size
        self.input_projection = nn.Conv2d(feature_dimension, hidden_dimension, 1)
        self.residual_stack = nn.Sequential(
            *(DecoderResidualBlock(hidden_dimension, normalization_groups) for block_index in range(residual_blocks))
        )
        decoder_blocks: list[nn.Module] = []
        input_channels = hidden_dimension
        for output_channels in upsample_channels:
            decoder_blocks.extend(
                (
                    DecoderUpsampleBlock(
                        input_channels,
                        output_channels,
                        normalization_groups,
                    ),
                    DecoderResidualBlock(output_channels, normalization_groups),
                )
            )
            input_channels = output_channels
        self.upsampler = nn.Sequential(*decoder_blocks)
        self.output_head = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(input_channels, 3, 3, padding=1),
            nn.Sigmoid(),
        )

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        latents: Float[Tensor, "batch time patch feature"],
    ) -> Float[Tensor, "batch time channel height width"]:
        if latents.shape[2] != self.patch_grid_size**2:
            raise ValueError(f"Expected {self.patch_grid_size**2} DINO patch tokens")
        feature_maps = rearrange(
            latents,
            "batch time (height width) feature -> (batch time) feature height width",
            height=self.patch_grid_size,
            width=self.patch_grid_size,
        )
        images = self.output_head(self.upsampler(self.residual_stack(self.input_projection(feature_maps))))
        return rearrange(
            images,
            "(batch time) channel height width -> batch time channel height width",
            batch=latents.shape[0],
            time=latents.shape[1],
        )
