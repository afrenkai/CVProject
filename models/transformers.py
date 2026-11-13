from collections.abc import Callable
import torch
from beartype import beartype
from diffusers.models.embeddings import get_timestep_embedding
from einops import rearrange
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.attention.flex_attention import BlockMask, flex_attention
from models.model_types import AttentionBackend, TokenLayout


# vaswani PE, stole from old code.
def sinusoidal_timestep_embedding(timesteps: Tensor, dimension: int) -> Tensor:
    return get_timestep_embedding(
        timesteps.reshape(-1).float(),
        dimension,
        flip_sin_to_cos=True,
        downscale_freq_shift=min(1, dimension // 2 - 1),
    )


@jaxtyped(typechecker=beartype)
def frame_position_embeddings(
    reference: Float[Tensor, "..."], dimension: int, start: int, stop: int
) -> Float[Tensor, "position hidden"]:
    return sinusoidal_timestep_embedding(
        torch.arange(start, stop, device=reference.device, dtype=reference.dtype),
        dimension,
    )


def by_modality(
    grid: Tensor,
    layout: TokenLayout | None,
    video_module: Callable[[Tensor], Tensor],
    audio_module: Callable[[Tensor], Tensor] | None,
    fusion_module: Callable[[Tensor], Tensor] | None = None,
) -> Tensor:
    if layout is None or audio_module is None:
        return video_module(grid)
    video_tokens, audio_tokens = layout[0], layout[1]
    parts = [
        video_module(grid[:, :, :video_tokens]),
        audio_module(grid[:, :, video_tokens : video_tokens + audio_tokens]),
    ]
    if fusion_module is not None and grid.shape[2] > video_tokens + audio_tokens:
        parts.append(fusion_module(grid[:, :, video_tokens + audio_tokens :]))
    return torch.cat(parts, dim=2)


def expert_projections(dimension: int, head_dimension: int) -> nn.ModuleDict:
    return nn.ModuleDict(
        {
            "query": nn.Linear(dimension, dimension),
            "key": nn.Linear(dimension, dimension),
            "value": nn.Linear(dimension, dimension),
            "output": nn.Linear(dimension, dimension),
            "query_norm": nn.RMSNorm(head_dimension),
            "key_norm": nn.RMSNorm(head_dimension),
        }
    )


def expert_layers(dimension: int, mlp_ratio: float) -> nn.ModuleDict:
    modulation = nn.Sequential(nn.SiLU(), nn.Linear(dimension, dimension * 9))
    nn.init.zeros_(modulation[-1].weight)
    nn.init.zeros_(modulation[-1].bias)
    return nn.ModuleDict(
        {
            "feed_forward": SwiGLUFeedForward(dimension, mlp_ratio),
            "context_norm": nn.RMSNorm(dimension),
            "modulation": modulation,
        }
    )


# in general, referred to https://docs.pytorch.org/docs/master/generated/torch.nn.MultiheadAttention.html for this
class MultiHeadAttention(nn.Module):
    def __init__(
        self, dimension: int, heads: int, attention_backend: AttentionBackend, layout: TokenLayout | None = None
    ) -> None:
        super().__init__()
        assert dimension % heads == 0
        self.attention_backend = attention_backend
        self.flex_attention = torch.compile(flex_attention, dynamic=False)
        self.heads = heads
        self.head_dimension = dimension // heads
        self.layout = layout
        self.query_projection = nn.Linear(dimension, dimension)
        self.key_projection = nn.Linear(dimension, dimension)
        self.value_projection = nn.Linear(dimension, dimension)
        self.query_norm = nn.RMSNorm(self.head_dimension)
        self.key_norm = nn.RMSNorm(self.head_dimension)
        self.output_projection = nn.Linear(dimension, dimension)
        self.audio_projections = None if layout is None else expert_projections(dimension, self.head_dimension)
        self.fusion_projections = (
            None if layout is None or layout[2] == 0 else expert_projections(dimension, self.head_dimension)
        )

    def normalized_heads(self, tokens: Tensor, norm: nn.Module) -> Tensor:
        return rearrange(
            norm(rearrange(tokens, "... (head channel) -> ... head channel", head=self.heads)),
            "... head channel -> ... (head channel)",
        )

    def projector(self, projection: nn.Module, norm: nn.Module | None) -> Callable[[Tensor], Tensor]:
        return lambda tokens: projection(tokens) if norm is None else self.normalized_heads(projection(tokens), norm)

    def expert(
        self, projections: nn.ModuleDict | None, name: str, normalized: bool
    ) -> Callable[[Tensor], Tensor] | None:
        if projections is None:
            return None
        return self.projector(projections[name], projections[f"{name}_norm"] if normalized else None)

    def project(self, grid: Tensor, name: str, projection: nn.Linear, norm: nn.Module | None) -> Tensor:
        normalized = norm is not None
        return by_modality(
            grid,
            self.layout,
            self.projector(projection, norm),
            self.expert(self.audio_projections, name, normalized),
            self.expert(self.fusion_projections, name, normalized),
        )

    def history_heads(self, history: Tensor, name: str, projection: nn.Linear, norm: nn.Module | None) -> Tensor:
        expert = self.expert(self.audio_projections, name, norm is not None) or self.projector(projection, norm)
        return rearrange(expert(history), "batch token (head channel) -> batch head token channel", head=self.heads)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor | BlockMask | None = None,
        history: Tensor | None = None,
    ) -> Tensor:
        query_tokens, key_tokens, value_tokens = (
            rearrange(tokens, "batch time token (head channel) -> batch head (time token) channel", head=self.heads)
            for tokens in (
                self.project(query, "query", self.query_projection, self.query_norm),
                self.project(key, "key", self.key_projection, self.key_norm),
                self.project(value, "value", self.value_projection, None),
            )
        )
        if history is not None:
            key_tokens = torch.cat(
                (key_tokens, self.history_heads(history, "key", self.key_projection, self.key_norm)), dim=2
            )
            value_tokens = torch.cat(
                (value_tokens, self.history_heads(history, "value", self.value_projection, None)), dim=2
            )
        if (
            self.attention_backend == "sdpa"
            or (self.attention_backend == "flex" and attention_mask is None)
            or isinstance(attention_mask, Tensor)
        ):
            attended = F.scaled_dot_product_attention(query_tokens, key_tokens, value_tokens, attn_mask=attention_mask)
        else:
            attended = self.flex_attention(
                query_tokens.to(value_tokens.dtype),
                key_tokens.to(value_tokens.dtype),
                value_tokens,
                block_mask=attention_mask,
                kernel_options={"BACKEND": "FLASH" if self.attention_backend == "flex_flash" else "TRITON"},
            )
        grid = rearrange(
            attended, "batch head (time token) channel -> batch time token (head channel)", token=query.shape[2]
        )
        return self.project(grid, "output", self.output_projection, None)


class SwiGLUFeedForward(nn.Module):
    def __init__(self, dimension: int, mlp_ratio: float) -> None:
        super().__init__()
        hidden_dimension = int(2 * dimension * mlp_ratio / 3)
        self.input_projection = nn.Linear(dimension, hidden_dimension * 2)
        self.output_projection = nn.Linear(hidden_dimension, dimension)

    def forward(self, tokens: Tensor) -> Tensor:
        values, gates = self.input_projection(tokens).chunk(2, dim=-1)
        return self.output_projection(values * F.silu(gates))


class AdaptiveAttentionBlock(nn.Module):
    def __init__(
        self,
        dimension: int,
        heads: int,
        mlp_ratio: float,
        attention_backend: AttentionBackend,
        layout: TokenLayout | None = None,
    ) -> None:
        super().__init__()
        self.layout = layout
        self.self_attention = MultiHeadAttention(dimension, heads, attention_backend, layout)
        self.cross_attention = MultiHeadAttention(dimension, heads, attention_backend, layout)
        self.feed_forward = SwiGLUFeedForward(dimension, mlp_ratio)
        self.self_norm = nn.RMSNorm(dimension, elementwise_affine=False)
        self.cross_norm = nn.RMSNorm(dimension, elementwise_affine=False)
        self.feed_forward_norm = nn.RMSNorm(dimension, elementwise_affine=False)
        self.context_norm = nn.RMSNorm(dimension)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dimension, dimension * 9))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)
        self.audio_layers = None if layout is None else expert_layers(dimension, mlp_ratio)
        self.fusion_layers = None if layout is None or layout[2] == 0 else expert_layers(dimension, mlp_ratio)

    def split(self, name: str) -> tuple[Callable[[Tensor], Tensor] | None, Callable[[Tensor], Tensor] | None]:
        return (
            None if self.audio_layers is None else self.audio_layers[name],
            None if self.fusion_layers is None else self.fusion_layers[name],
        )

    def to_grid(self, tokens: Tensor, fused: bool) -> Tensor:
        if self.layout is None:
            return rearrange(tokens, "batch token hidden -> batch 1 token hidden")
        step = sum(self.layout) if fused else self.layout[0] + self.layout[1]
        return rearrange(tokens, "batch (time token) hidden -> batch time token hidden", token=step)

    def forward(
        self,
        targets: Tensor,
        context: Tensor,
        condition: Tensor,
        attention_mask: Tensor | BlockMask,
        context_mask: Tensor | BlockMask | None = None,
        history: Tensor | None = None,
    ) -> Tensor:
        grid = self.to_grid(targets, fused=True)
        context_grid = self.to_grid(context, fused=False)
        if condition.shape[1] == 1:
            condition_grid = rearrange(condition, "batch 1 hidden -> batch 1 1 hidden")
            if self.layout is not None:
                condition_grid = condition_grid.expand(-1, -1, grid.shape[2], -1)
        else:
            condition_grid = self.to_grid(condition, fused=True)
        (
            self_shift,
            self_scale,
            self_gate,
            cross_shift,
            cross_scale,
            cross_gate,
            feed_forward_shift,
            feed_forward_scale,
            feed_forward_gate,
        ) = by_modality(condition_grid, self.layout, self.modulation, *self.split("modulation")).chunk(9, dim=-1)
        self_input = self.self_norm(grid) * (1 + self_scale) + self_shift
        grid = grid + self_gate * self.self_attention(self_input, self_input, self_input, attention_mask)
        cross_input = self.cross_norm(grid) * (1 + cross_scale) + cross_shift
        normalized_context = by_modality(context_grid, self.layout, self.context_norm, *self.split("context_norm"))
        history_norm = self.context_norm if self.audio_layers is None else self.audio_layers["context_norm"]
        normalized_history = None if history is None else history_norm(history)
        grid = grid + cross_gate * self.cross_attention(
            cross_input, normalized_context, normalized_context, context_mask, normalized_history
        )
        feed_forward_input = self.feed_forward_norm(grid) * (1 + feed_forward_scale) + feed_forward_shift
        grid = grid + feed_forward_gate * by_modality(
            feed_forward_input, self.layout, self.feed_forward, *self.split("feed_forward")
        )
        return rearrange(grid, "batch time token hidden -> batch (time token) hidden")
