import torch
from beartype import beartype
from diffusers.models.embeddings import get_timestep_embedding
from einops import rearrange
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.attention.flex_attention import BlockMask, flex_attention
from models.model_types import AttentionBackend


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


# in general, referred to https://docs.pytorch.org/docs/master/generated/torch.nn.MultiheadAttention.html for this
class MultiHeadAttention(nn.Module):
    def __init__(self, dimension: int, heads: int, attention_backend: AttentionBackend) -> None:
        super().__init__()
        assert dimension % heads == 0
        self.attention_backend = attention_backend
        self.flex_attention = torch.compile(flex_attention, dynamic=False)
        self.heads = heads
        self.head_dimension = dimension // heads
        self.query_projection = nn.Linear(dimension, dimension)
        self.key_projection = nn.Linear(dimension, dimension)
        self.value_projection = nn.Linear(dimension, dimension)
        self.query_norm = nn.RMSNorm(self.head_dimension)
        self.key_norm = nn.RMSNorm(self.head_dimension)
        self.output_projection = nn.Linear(dimension, dimension)

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor, attention_mask: Tensor | BlockMask | None = None
    ) -> Tensor:
        query_tokens = self.query_norm(
            rearrange(
                self.query_projection(query), "batch query (head channel) -> batch head query channel", head=self.heads
            )
        )
        key_tokens = self.key_norm(
            rearrange(self.key_projection(key), "batch key (head channel) -> batch head key channel", head=self.heads)
        )
        value_tokens = rearrange(
            self.value_projection(value), "batch key (head channel) -> batch head key channel", head=self.heads
        )
        if self.attention_backend == "sdpa" or (self.attention_backend == "flex" and attention_mask is None):
            attended = F.scaled_dot_product_attention(query_tokens, key_tokens, value_tokens, attn_mask=attention_mask)
        else:
            attended = self.flex_attention(
                query_tokens.to(value_tokens.dtype),
                key_tokens.to(value_tokens.dtype),
                value_tokens,
                block_mask=attention_mask,
                kernel_options={"BACKEND": "FLASH" if self.attention_backend == "flex_flash" else "TRITON"},
            )
        return self.output_projection(rearrange(attended, "batch head query channel -> batch query (head channel)"))


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
    def __init__(self, dimension: int, heads: int, mlp_ratio: float, attention_backend: AttentionBackend) -> None:
        super().__init__()
        self.self_attention = MultiHeadAttention(dimension, heads, attention_backend)
        self.cross_attention = MultiHeadAttention(dimension, heads, attention_backend)
        self.feed_forward = SwiGLUFeedForward(dimension, mlp_ratio)
        self.self_norm = nn.RMSNorm(dimension, elementwise_affine=False)
        self.cross_norm = nn.RMSNorm(dimension, elementwise_affine=False)
        self.feed_forward_norm = nn.RMSNorm(dimension, elementwise_affine=False)
        self.context_norm = nn.RMSNorm(dimension)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dimension, dimension * 9))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(
        self, targets: Tensor, context: Tensor, condition: Tensor, attention_mask: Tensor | BlockMask
    ) -> Tensor:
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
        ) = self.modulation(condition).chunk(9, dim=-1)
        self_input = self.self_norm(targets) * (1 + self_scale) + self_shift
        targets = targets + self_gate * self.self_attention(self_input, self_input, self_input, attention_mask)
        cross_input = self.cross_norm(targets) * (1 + cross_scale) + cross_shift
        normalized_context = self.context_norm(context)
        targets = targets + cross_gate * self.cross_attention(cross_input, normalized_context, normalized_context)
        feed_forward_input = self.feed_forward_norm(targets) * (1 + feed_forward_scale) + feed_forward_shift
        return targets + feed_forward_gate * self.feed_forward(feed_forward_input)
