from typing import Literal
import torch
from beartype import beartype
from einops import einsum, rearrange
from jaxtyping import Float, Int, jaxtyped
from pydantic import BaseModel, ConfigDict, Field, PositiveInt
from torch import Tensor, nn
from torch.nn import functional as F
from deprecated.models.kda import KimiDeltaAttention
from models.transformers import (
    MultiHeadAttention,
    SwiGLUFeedForward,
    frame_position_embeddings,
)


class HybridHistoryConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    block_pattern: tuple[Literal["kda", "softmax"], ...] = ("kda", "kda", "kda", "softmax")
    chunk_size: PositiveInt = 16
    minimum_log_decay: float = Field(default=-5.0, lt=0.0)


class LinearAttention(nn.Module):
    def __init__(self, dimension: int, heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if dimension % heads:
            raise ValueError("Attention dimension must be divisible by its head count")
        self.heads = heads
        self.head_dimension = dimension // heads
        self.query_projection = nn.Linear(dimension, dimension, bias=False)
        self.key_projection = nn.Linear(dimension, dimension, bias=False)
        self.value_projection = nn.Linear(dimension, dimension, bias=False)
        self.output_projection = nn.Linear(dimension, dimension, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        causal_prefix_lengths: Tensor | None = None,
    ) -> Tensor:
        query_features = (
            F.elu(
                rearrange(
                    self.query_projection(query),
                    "batch token (head channel) -> batch head token channel",
                    head=self.heads,
                )
            )
            + 1.0
        )
        key_features = (
            F.elu(
                rearrange(
                    self.key_projection(key),
                    "batch token (head channel) -> batch head token channel",
                    head=self.heads,
                )
            )
            + 1.0
        )
        values = rearrange(
            self.value_projection(value),
            "batch token (head channel) -> batch head token channel",
            head=self.heads,
        )

        if causal_prefix_lengths is None:
            key_sum = key_features.sum(dim=2, keepdim=True)
            key_values = einsum(
                key_features,
                values,
                "batch head time key, batch head time value -> batch head key value",
            )
            attended = einsum(
                query_features,
                key_values,
                "batch head query key, batch head key value -> batch head query value",
            )
            normalizer = einsum(
                query_features,
                key_sum,
                "batch head query key, batch head singleton key -> batch head query singleton",
            )
        else:
            if causal_prefix_lengths.ndim != 1 or causal_prefix_lengths.shape[0] != query.shape[1]:
                raise ValueError("Causal prefix lengths must provide one value per query token")
            if int(causal_prefix_lengths.min()) < 1 or int(causal_prefix_lengths.max()) > key.shape[1]:
                raise ValueError("Causal prefix lengths fall outside the key sequence")
            prefix_indices = causal_prefix_lengths.to(device=query.device, dtype=torch.long) - 1
            cumulative_keys = key_features.cumsum(dim=2)
            cumulative_key_values = einsum(
                key_features,
                values,
                "batch head time key, batch head time value -> batch head time key value",
            ).cumsum(dim=2)
            selected_keys = cumulative_keys[:, :, prefix_indices]
            selected_key_values = cumulative_key_values[:, :, prefix_indices]
            attended = einsum(
                query_features,
                selected_key_values,
                "batch head query key, batch head query key value -> batch head query value",
            )
            normalizer = einsum(
                query_features,
                selected_keys,
                "batch head query key, batch head query key -> batch head query",
            )[:, :, :, None]

        attended = attended / normalizer.clamp_min(1e-6)
        attended = rearrange(attended, "batch head token channel -> batch token (head channel)")
        return self.output_projection(self.dropout(attended))


class HistoryMixerBlock(nn.Module):
    def __init__(
        self,
        dimension: int,
        mlp_ratio: float,
        dropout: float,
        attention: LinearAttention | KimiDeltaAttention | MultiHeadAttention,
    ) -> None:
        super().__init__()
        self.attention_norm = nn.RMSNorm(dimension)
        self.attention = attention
        self.feed_forward_norm = nn.RMSNorm(dimension)
        self.feed_forward = SwiGLUFeedForward(dimension, mlp_ratio, dropout)
        self.dropout = nn.Dropout(dropout)

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        tokens: Float[Tensor, "batch time hidden"],
        causal_prefix_lengths: Int[Tensor, "time"],  # noqa: F821
    ) -> Float[Tensor, "batch time hidden"]:
        normalized = self.attention_norm(tokens)
        if isinstance(self.attention, LinearAttention):
            attended = self.attention(normalized, normalized, normalized, causal_prefix_lengths)
        elif isinstance(self.attention, KimiDeltaAttention):
            attended = self.attention(normalized)[0]
        else:
            positions = torch.arange(tokens.shape[1], device=tokens.device)
            attention_mask = positions[None, :] < causal_prefix_lengths[:, None]
            attended = self.attention(normalized, normalized, normalized, attention_mask=attention_mask)
        tokens = tokens + self.dropout(attended)
        return tokens + self.dropout(self.feed_forward(self.feed_forward_norm(tokens)))


class AudioVisualHistoryMixer(nn.Module):
    def __init__(
        self,
        video_dimension: int,
        audio_dimension: int,
        action_dimension: int,
        hidden_dimension: int,
        heads: int,
        layers: int,
        mlp_ratio: float,
        dropout: float,
        config: HybridHistoryConfig | None = None,
    ) -> None:
        if config is not None and (layers < 1 or layers % len(config.block_pattern)):
            raise ValueError("Hybrid history requires complete three-KDA/one-softmax cycles")
        super().__init__()
        self.hidden_dimension = hidden_dimension
        self.video_projection = nn.Linear(video_dimension, hidden_dimension)
        self.audio_projection = nn.Linear(audio_dimension, hidden_dimension)
        self.action_projection = nn.Linear(action_dimension, hidden_dimension, bias=False)
        self.modality_norm = nn.RMSNorm(hidden_dimension)
        self.blocks = nn.ModuleList()
        for layer_index in range(layers):
            if config is None:
                attention = LinearAttention(hidden_dimension, heads, dropout)
            elif config.block_pattern[layer_index % len(config.block_pattern)] == "kda":
                attention = KimiDeltaAttention(
                    hidden_dimension, heads, hidden_dimension // heads, config.chunk_size, config.minimum_log_decay
                )
            else:
                attention = MultiHeadAttention(hidden_dimension, heads, dropout)
            self.blocks.append(HistoryMixerBlock(hidden_dimension, mlp_ratio, dropout, attention))

    def frame_tokens(
        self,
        video: Tensor,
        audio: Tensor,
        action_embeddings: Tensor,
        frame_start: int,
    ) -> Tensor:
        video_state = self.video_projection(video).mean(dim=2)
        audio_state = self.audio_projection(audio)
        positions = frame_position_embeddings(video, self.hidden_dimension, frame_start, frame_start + video.shape[1])
        position_state = rearrange(positions, "time hidden -> 1 time hidden")
        return self.modality_norm(video_state + audio_state + action_embeddings + position_state)

    def forward(
        self,
        context_video: Tensor,
        context_audio: Tensor,
        target_video: Tensor,
        target_audio: Tensor,
        target_actions: Tensor,
        context_actions: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        context_steps = context_video.shape[1]
        context_tokens = self.frame_tokens(
            context_video,
            context_audio,
            self.context_action_embeddings(context_video, context_actions),
            -context_steps,
        )
        target_tokens = self.frame_tokens(target_video, target_audio, self.action_projection(target_actions), 0)
        tokens = torch.cat((context_tokens, target_tokens), dim=1)
        causal_prefix_lengths = torch.arange(1, tokens.shape[1] + 1, device=tokens.device)
        for block in self.blocks:
            tokens = block(tokens, causal_prefix_lengths)
        return tokens[:, context_steps:], tokens[:, context_steps - 1]

    @jaxtyped(typechecker=beartype)
    def context_action_embeddings(
        self,
        context_video: Float[Tensor, "batch context_time patch video_feature"],
        context_actions: Float[Tensor, "batch context_time action"] | None,
    ) -> Float[Tensor, "batch context_time hidden"]:
        if context_actions is None:
            return context_video.new_zeros((*context_video.shape[:2], self.hidden_dimension))
        if context_actions.shape != (
            *context_video.shape[:2],
            self.action_projection.in_features,
        ):
            raise ValueError("Previous executed actions must align with observation history")
        return self.action_projection(context_actions)

    @jaxtyped(typechecker=beartype)
    def encode_context(
        self,
        context_video: Float[Tensor, "batch context_time patch video_feature"],
        context_audio: Float[Tensor, "batch context_time audio_feature"],
        context_actions: Float[Tensor, "batch context_time action"] | None = None,
    ) -> Float[Tensor, "batch hidden"]:
        tokens = self.frame_tokens(
            context_video,
            context_audio,
            self.context_action_embeddings(context_video, context_actions),
            -context_video.shape[1],
        )
        causal_prefix_lengths = torch.arange(1, tokens.shape[1] + 1, device=tokens.device)
        for block in self.blocks:
            tokens = block(tokens, causal_prefix_lengths)
        return tokens[:, -1]
