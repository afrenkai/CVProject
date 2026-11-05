import torch
from beartype import beartype
from diffusers.models.embeddings import get_timestep_embedding
from einops import rearrange
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn
from torch.distributed import ProcessGroup
from torch.nn import functional as F
from torch.nn.attention.flex_attention import BlockMask, flex_attention
from models.sequence_parallel import heads_to_sequence, sequence_to_heads


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
    def __init__(
        self,
        dimension: int,
        heads: int,
        dropout: float = 0.0,
        key_value_heads: int | None = None,
        attention_backend: str = "sdpa",
    ) -> None:
        super().__init__()
        if dimension % heads:
            raise ValueError("Attention dimension must be divisible by its head count")
        if attention_backend not in {"sdpa", "flex", "flex_flash"}:
            raise ValueError("Unknown attention backend")
        if attention_backend != "sdpa" and dropout > 0.0:
            raise ValueError("Flex attention does not support attention dropout")
        self.attention_backend = attention_backend
        self.flex_attention = torch.compile(flex_attention, dynamic=False)
        self.sequence_group: ProcessGroup | None = None
        self.heads = heads
        self.key_value_heads = key_value_heads or heads
        if heads % self.key_value_heads:
            raise ValueError("Query heads must be a multiple of key-value heads")
        self.head_dimension = dimension // heads
        self.dropout = dropout
        self.query_projection = nn.Linear(dimension, dimension)
        self.key_projection = nn.Linear(dimension, self.key_value_heads * self.head_dimension)
        self.value_projection = nn.Linear(dimension, self.key_value_heads * self.head_dimension)
        self.query_norm = nn.RMSNorm(self.head_dimension)
        self.key_norm = nn.RMSNorm(self.head_dimension)
        self.output_projection = nn.Linear(dimension, dimension)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor | BlockMask | None = None,
    ) -> Tensor:
        """
        qkv are (3,...) tensors of shape bxqxL where L is seq_len
        need to expand into bxhxqxch so can operate on the channel dim
        see query_proj method, key_norm and value proj
        still uses F.SDPA (with option for attn mask) and returns bxqxL expanded in

        @param query: query tensorfor self-attn
        @param key: key tensor for self-attn
        @param value: value tensor for self-attn
        @param attn_mask: optional attention mask to hide future preds from current state. written to be compat with F.SDPA
        Returns:
            O = A(QK^T/sqrt(d_h)V) which is of shap e b x q x L (L = hxch)

        """
        query_tokens = self.query_norm(
            rearrange(
                self.query_projection(query),
                "batch query (head channel) -> batch head query channel",
                head=self.heads,
            )
        )
        key_tokens = self.key_norm(
            rearrange(
                self.key_projection(key),
                "batch key (head channel) -> batch head key channel",
                head=self.key_value_heads,
            )
        )
        value_tokens = rearrange(
            self.value_projection(value),
            "batch key (head channel) -> batch head key channel",
            head=self.key_value_heads,
        )
        if self.sequence_group is not None:
            query_tokens = sequence_to_heads(query_tokens, self.sequence_group)
            key_tokens = sequence_to_heads(key_tokens, self.sequence_group)
            value_tokens = sequence_to_heads(value_tokens, self.sequence_group)
        if self.attention_backend == "sdpa" or (self.attention_backend == "flex" and attention_mask is None):
            attended = F.scaled_dot_product_attention(
                query_tokens,
                key_tokens,
                value_tokens,
                attn_mask=attention_mask,
                dropout_p=self.dropout if self.training else 0.0,
                enable_gqa=self.key_value_heads != self.heads,
            )
        else:
            attended = self.flex_attention(
                query_tokens.to(value_tokens.dtype),
                key_tokens.to(value_tokens.dtype),
                value_tokens,
                block_mask=attention_mask,
                enable_gqa=self.key_value_heads != self.heads,
                kernel_options={"BACKEND": "FLASH" if self.attention_backend == "flex_flash" else "TRITON"},
            )
        if self.sequence_group is not None:
            attended = heads_to_sequence(attended, self.sequence_group)
        attended = rearrange(
            attended,
            "batch head query channel -> batch query (head channel)",
        )
        return self.output_projection(attended)


class SwiGLUFeedForward(nn.Module):
    def __init__(self, dimension: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        hidden_dimension = int(2 * dimension * mlp_ratio / 3)
        self.input_projection = nn.Linear(dimension, hidden_dimension * 2)
        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(hidden_dimension, dimension)

    def forward(self, tokens: Tensor) -> Tensor:
        values, gates = self.input_projection(tokens).chunk(2, dim=-1)
        return self.output_projection(self.dropout(values * F.silu(gates)))


class AdaptiveAttentionBlock(nn.Module):
    def __init__(
        self,
        dimension: int,
        heads: int,
        mlp_ratio: float,
        dropout: float,
        key_value_heads: int | None = None,
        attention_backend: str = "sdpa",
    ) -> None:
        super().__init__()
        self.self_attention = MultiHeadAttention(dimension, heads, dropout, key_value_heads, attention_backend)
        self.cross_attention = MultiHeadAttention(dimension, heads, dropout, key_value_heads, attention_backend)
        self.feed_forward = SwiGLUFeedForward(dimension, mlp_ratio, dropout)
        self.self_norm = nn.RMSNorm(dimension, elementwise_affine=False)
        self.cross_norm = nn.RMSNorm(dimension, elementwise_affine=False)
        self.feed_forward_norm = nn.RMSNorm(dimension, elementwise_affine=False)
        self.context_norm = nn.RMSNorm(dimension)
        self.dropout = nn.Dropout(dropout)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dimension, dimension * 9))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(
        self,
        targets: Tensor,
        context: Tensor,
        condition: Tensor,
        attention_mask: Tensor | BlockMask,
    ) -> Tensor:
        modulation_values = self.modulation(condition).chunk(9, dim=-1)
        self_shift, self_scale, self_gate = modulation_values[:3]
        cross_shift, cross_scale, cross_gate = modulation_values[3:6]
        feed_forward_shift, feed_forward_scale, feed_forward_gate = modulation_values[6:]

        self_input = self.self_norm(targets) * (1 + self_scale) + self_shift
        self_output = self.self_attention(
            self_input,
            self_input,
            self_input,
            attention_mask=attention_mask,
        )
        targets = targets + self.dropout(self_gate * self_output)

        cross_input = self.cross_norm(targets) * (1 + cross_scale) + cross_shift
        normalized_context = self.context_norm(context)
        cross_output = self.cross_attention(
            cross_input,
            normalized_context,
            normalized_context,
        )
        targets = targets + self.dropout(cross_gate * cross_output)

        feed_forward_input = self.feed_forward_norm(targets) * (1 + feed_forward_scale) + feed_forward_shift
        feed_forward_output = self.feed_forward(feed_forward_input)
        return targets + self.dropout(feed_forward_gate * feed_forward_output)
