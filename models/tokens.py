import torch
from beartype import beartype
from diffusers.models.embeddings import get_2d_sincos_pos_embed
from einops import rearrange
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn
from models.model_types import AudioTokenConfig
from models.transformers import (
    frame_position_embeddings,
    sinusoidal_timestep_embedding,
)


class AudioVisualTokenProjector(nn.Module):
    def __init__(
        self,
        video_dimension: int,
        audio_dimension: int,
        hidden_dimension: int,
        spatial_grid: tuple[int, int] | None,
        audio_tokens: AudioTokenConfig,
        fusion_tokens: int = 0,
    ) -> None:
        super().__init__()
        self.audio_tokens = audio_tokens
        assert audio_dimension % self.audio_tokens.tokens_per_step == 0
        self.video_projection = nn.Linear(video_dimension, hidden_dimension)
        self.audio_projection = nn.Linear(audio_dimension // self.audio_tokens.tokens_per_step, hidden_dimension)
        self.modality_embedding = nn.Embedding(2, hidden_dimension)
        self.spatial_grid = spatial_grid
        self.hidden_dimension = hidden_dimension
        self.fusion_queries = (
            None if fusion_tokens == 0 else nn.Parameter(torch.randn(fusion_tokens, hidden_dimension) * 0.02)
        )
        self.register_buffer(
            "spatial_positions",
            (
                None
                if spatial_grid is None
                else get_2d_sincos_pos_embed(hidden_dimension, spatial_grid, output_type="pt").float()
            ),
            persistent=False,
        )
        channel_positions = sinusoidal_timestep_embedding(torch.arange(self.audio_tokens.channels), hidden_dimension)
        self.register_buffer(
            "audio_channel_positions",
            channel_positions - channel_positions.mean(dim=0, keepdim=True),
            persistent=False,
        )
        part_positions = sinusoidal_timestep_embedding(torch.arange(self.audio_tokens.parts), hidden_dimension)
        self.register_buffer(
            "audio_part_positions",
            part_positions - part_positions.mean(dim=0, keepdim=True),
            persistent=False,
        )

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
            video_tokens = video_tokens + rearrange(
                self.spatial_positions.to(video), "patch hidden -> 1 1 patch hidden"
            )
        groups = [video_tokens, self.audio_grid(audio, frame_start, action_embeddings)]
        if self.fusion_queries is not None:
            groups.append(
                rearrange(self.fusion_queries.to(video_tokens), "token hidden -> 1 1 token hidden")
                + frame_positions
                + action_positions
            )
        return rearrange(torch.cat(groups, dim=2), "batch time token hidden -> batch (time token) hidden")

    @jaxtyped(typechecker=beartype)
    def history_tokens(
        self, audio: Float[Tensor, "batch time audio_feature"], frame_start: int
    ) -> Float[Tensor, "batch token_time hidden"]:
        actions = audio.new_zeros((*audio.shape[:2], self.hidden_dimension))
        return rearrange(
            self.audio_grid(audio, frame_start, actions), "batch time token hidden -> batch (time token) hidden"
        )

    def audio_grid(self, audio: Tensor, frame_start: int, action_embeddings: Tensor) -> Tensor:
        audio_frames = rearrange(
            audio,
            "batch time (channel phase part feature) -> batch time channel phase part feature",
            channel=self.audio_tokens.channels,
            phase=self.audio_tokens.frames_per_step,
            part=self.audio_tokens.parts,
        )
        audio_times = (
            torch.arange(
                frame_start * self.audio_tokens.frames_per_step,
                (frame_start + audio.shape[1]) * self.audio_tokens.frames_per_step,
                device=audio.device,
                dtype=audio.dtype,
            )
            / self.audio_tokens.frames_per_step
        )
        audio_positions = rearrange(
            sinusoidal_timestep_embedding(audio_times, self.hidden_dimension),
            "(time phase) hidden -> 1 time 1 phase 1 hidden",
            phase=self.audio_tokens.frames_per_step,
        )
        audio_tokens = (
            self.audio_projection(audio_frames)
            + self.modality_embedding.weight[1]
            + audio_positions
            + rearrange(self.audio_channel_positions, "channel hidden -> 1 1 channel 1 1 hidden")
            + rearrange(self.audio_part_positions, "part hidden -> 1 1 1 1 part hidden")
            + rearrange(action_embeddings, "batch time hidden -> batch time 1 1 1 hidden")
        )
        return rearrange(audio_tokens, "batch time channel phase part hidden -> batch time (channel phase part) hidden")
