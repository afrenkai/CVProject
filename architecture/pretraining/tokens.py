import torch
from einops import rearrange
from torch import Tensor, nn


class AudioVisualTokenProjector(nn.Module):
    def __init__(
        self,
        video_dimension: int,
        audio_dimension: int,
        hidden_dimension: int,
    ) -> None:
        super().__init__()
        self.video_projection = nn.Linear(video_dimension, hidden_dimension)
        self.audio_projection = nn.Linear(audio_dimension, hidden_dimension)
        self.modality_embedding = nn.Embedding(2, hidden_dimension)

    def forward(
        self,
        video: Tensor,
        audio: Tensor,
        frame_embeddings: Tensor,
        action_embeddings: Tensor,
    ) -> Tensor:
        frame_positions = rearrange(
            frame_embeddings,
            "time hidden -> 1 time 1 hidden",
        )
        action_positions = rearrange(
            action_embeddings,
            "batch time hidden -> batch time 1 hidden",
        )
        video_tokens = (
            self.video_projection(video)
            + self.modality_embedding.weight[0]
            + frame_positions
            + action_positions
        )
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
