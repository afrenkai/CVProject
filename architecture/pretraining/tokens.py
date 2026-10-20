import torch
from beartype import beartype
from diffusers.models.embeddings import get_2d_sincos_pos_embed
from einops import rearrange
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn

from architecture.pretraining.transformers import frame_position_embeddings


class AudioVisualTokenProjector(nn.Module):
    """
    spatial affine fn, effectively projects from d_v, d_a -> E[1...k]

    """
    def __init__(
        self,
        video_dimension: int,
        audio_dimension: int,
        hidden_dimension: int,
        spatial_grid: tuple[int, int] | None = None,
    ) -> None:
        super().__init__()
        self.video_projection = nn.Linear(video_dimension, hidden_dimension)
        self.audio_projection = nn.Linear(audio_dimension, hidden_dimension)
        self.modality_embedding = nn.Embedding(2, hidden_dimension)
        self.spatial_grid = spatial_grid
        self.hidden_dimension = hidden_dimension
        self.register_buffer("spatial_positions", None, persistent=False)

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        video: Float[Tensor, "batch time patch video_feature"],
        audio: Float[Tensor, "batch time audio_feature"],
        frame_start: int,
        action_embeddings: Float[Tensor, "batch time hidden"] | None = None,
    ) -> Float[Tensor, "batch token_time hidden"]:
        frame_positions = rearrange(
            frame_position_embeddings(video, self.hidden_dimension, frame_start, frame_start + video.shape[1]),
            "time hidden -> 1 time 1 hidden",
        )
        if action_embeddings is None:
            action_embeddings = video.new_zeros((*video.shape[:2], self.hidden_dimension))
        action_positions = rearrange(
            action_embeddings,
            "batch time hidden -> batch time 1 hidden",
        )
        video_tokens = (
            self.video_projection(video) + self.modality_embedding.weight[0] + frame_positions + action_positions
        )
        if self.spatial_grid is not None:
            if video.shape[2] != self.spatial_grid[0] * self.spatial_grid[1]:
                raise ValueError("Video patches do not match the configured spatial grid")
            if self.spatial_positions is None:
                self.spatial_positions = get_2d_sincos_pos_embed(
                    self.hidden_dimension,
                    self.spatial_grid,
                    device=video.device,
                    output_type="pt",
                ).to(video)
            video_tokens = video_tokens + rearrange(self.spatial_positions, "patch hidden -> 1 1 patch hidden")
        audio_tokens = (
            self.audio_projection(audio)
            + self.modality_embedding.weight[1]
            + frame_positions[:, :, 0]
            + action_positions[:, :, 0]
        )
        return rearrange(
            torch.cat((video_tokens, audio_tokens[:, :, None]), dim=2),
            "batch time token hidden -> batch (time token) hidden",
        )
