import math

import torch
from beartype import beartype
from einops import einsum, rearrange
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn
from torch.nn import functional as F


@jaxtyped(typechecker=beartype)
def chunk_kimi_delta_attention(
    queries: Float[Tensor, "batch time head key"],
    keys: Float[Tensor, "batch time head key"],
    values: Float[Tensor, "batch time head value"],
    log_decay: Float[Tensor, "batch time head key"],
    write_strength: Float[Tensor, "batch time head"],
    initial_state: Float[Tensor, "batch head key value"] | None = None,
    chunk_size: int = 16,
) -> tuple[
    Float[Tensor, "batch time head value"],
    Float[Tensor, "batch head key value"],
]:
    batch_size, time_steps, heads, key_dimension = queries.shape
    value_dimension = values.shape[-1]
    if time_steps == 0 or chunk_size <= 0:
        raise ValueError("KDA requires tokens and a positive chunk size")

    # state has to be bxhxkxv
    expected_state_shape = (
        batch_size,
        heads,
        key_dimension,
        value_dimension,
    )
    # little check for device across torch since I don't want to wantonly cast to and across devices, esp since we have DDP for n=2
    invalid_state = initial_state is not None and (
        initial_state.shape != expected_state_shape
        or initial_state.device != queries.device
    )
    if invalid_state:
        raise ValueError(
            f"Expected KDA state {expected_state_shape} on {queries.device}"
        )

    state = (
        queries.new_zeros(expected_state_shape, dtype=torch.float32)
        if initial_state is None
        else initial_state.float()
    )
    output_chunks: list[Tensor] = []
    for chunk_start in range(0, time_steps, chunk_size):
        chunk_end = min(chunk_start + chunk_size, time_steps)
        chunk_queries = rearrange(
            queries[:, chunk_start:chunk_end],
            "batch chunk head key -> batch head chunk key",
        ).float() / math.sqrt(key_dimension)
        # little hack to do the /d_h here instead
        chunk_keys = rearrange(
            keys[:, chunk_start:chunk_end],
            "batch chunk head key -> batch head chunk key",
        ).float()
        chunk_values = rearrange(
            values[:, chunk_start:chunk_end],
            "batch chunk head value -> batch head chunk value",
        ).float()
        chunk_log_decay = rearrange(
            log_decay[:, chunk_start:chunk_end],
            "batch chunk head key -> batch head chunk key",
        ).float()
        chunk_write_strength = rearrange(
            write_strength[:, chunk_start:chunk_end],
            "batch chunk head -> batch head chunk",
        ).float()
        cumulative_log_decay = torch.cumsum(chunk_log_decay, dim=2)
        current_chunk_size = chunk_end - chunk_start
        # same as regular attn causal masking, i.e. for 3 tok (seq) at t=1:
        """
             | x -inf -inf
        Q    | x     x -inf
             | x    x    x

            K
        """
        # if tr, can do a tril like we do
        causal_mask = torch.ones(
            current_chunk_size,
            current_chunk_size,
            dtype=torch.bool,
            device=queries.device,
        ).tril()

        relative_log_decay = (
            cumulative_log_decay[:, :, :, None] - cumulative_log_decay[:, :, None]
        ).masked_fill(~causal_mask[None, None, :, :, None], 0.0)
        key_correlations = einsum(
            chunk_keys,
            chunk_keys,
            torch.exp(relative_log_decay),
            "batch head current key, batch head previous key, "
            "batch head current previous key -> batch head current previous",
        )
        strict_lower_correlations = torch.tril(key_correlations, diagonal=-1)
        identity = torch.eye(
            current_chunk_size, dtype=torch.float32, device=queries.device
        )
        triangular_system = (
            identity[None, None]
            + chunk_write_strength[..., None] * strict_lower_correlations
        )
        weighted_values = chunk_write_strength[..., None] * chunk_values
        weighted_keys = (
            chunk_write_strength[..., None]
            * torch.exp(cumulative_log_decay)
            * chunk_keys
        )

        # lol uh idk how to do this faster on our hardware
        transformed_values = torch.linalg.solve_triangular(
            triangular_system, weighted_values, upper=False
        )
        transformed_keys = torch.linalg.solve_triangular(
            triangular_system, weighted_keys, upper=False
        )

        within_chunk_attention = einsum(
            chunk_queries,
            chunk_keys,
            torch.exp(relative_log_decay),
            "batch head current key, batch head previous key, "
            "batch head current previous key -> batch head current previous",
        )
        within_chunk_attention = torch.tril(within_chunk_attention)
        pseudo_values = transformed_values - torch.matmul(transformed_keys, state)
        previous_state_output = einsum(
            chunk_queries * torch.exp(cumulative_log_decay),
            state,
            "batch head chunk key, batch head key value -> " "batch head chunk value",
        )
        output_chunks.append(
            previous_state_output + torch.matmul(within_chunk_attention, pseudo_values)
        )

        final_log_decay = cumulative_log_decay[:, :, -1]
        state = state * torch.exp(final_log_decay[..., None])
        keys_at_chunk_end = chunk_keys * torch.exp(
            final_log_decay[:, :, None] - cumulative_log_decay
        )
        state = state + einsum(
            keys_at_chunk_end,
            pseudo_values,
            "batch head chunk key, batch head chunk value -> batch head key value",
        )

    output = rearrange(
        torch.cat(output_chunks, dim=2),
        "batch head time value -> batch time head value",
    )
    return output.to(values.dtype), state


class KimiDeltaAttention(nn.Module):
    def __init__(
        self,
        hidden_dimension: int,
        heads: int,
        head_dimension: int,
        chunk_size: int = 16,
        minimum_log_decay: float = -5.0,
    ) -> None:
        super().__init__()
        invalid_configuration = (
            hidden_dimension <= 0
            or heads <= 0
            or head_dimension <= 0
            or chunk_size <= 0
            or minimum_log_decay >= 0.0
        )
        if invalid_configuration:
            raise ValueError(
                "KDA dimensions and chunk size must be positive; decay must be negative"
            )

        projection_dimension = heads * head_dimension
        self.heads = heads
        self.head_dimension = head_dimension
        self.chunk_size = chunk_size
        self.minimum_log_decay = minimum_log_decay
        self.query_projection = nn.Linear(
            hidden_dimension, projection_dimension, bias=False
        )
        self.key_projection = nn.Linear(
            hidden_dimension, projection_dimension, bias=False
        )
        self.value_projection = nn.Linear(
            hidden_dimension, projection_dimension, bias=False
        )
        self.decay_projection = nn.Sequential(
            nn.Linear(hidden_dimension, head_dimension, bias=False),
            nn.Linear(head_dimension, projection_dimension, bias=False),
        )
        self.write_projection = nn.Linear(hidden_dimension, heads, bias=False)
        self.output_gate_projection = nn.Linear(
            hidden_dimension, projection_dimension, bias=False
        )
        self.output_norm = nn.RMSNorm(head_dimension)
        self.output_projection = nn.Linear(
            projection_dimension, hidden_dimension, bias=False
        )

        self.log_decay_scale = nn.Parameter(torch.zeros(heads))
        time_step = torch.exp(
            torch.rand(heads, head_dimension) * (math.log(0.1) - math.log(0.001))
            + math.log(0.001)
        ).clamp(min=0.0001)
        self.decay_bias = nn.Parameter(time_step + torch.log(-torch.expm1(-time_step)))

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        tokens: Float[Tensor, "batch time hidden"],
        initial_state: Float[Tensor, "batch head state_key state_value"] | None = None,
    ) -> tuple[
        Float[Tensor, "batch time hidden"],
        Float[Tensor, "batch head state_key state_value"],
    ]:
        queries = rearrange(
            self.query_projection(tokens),
            "batch time (head channel) -> batch time head channel",
            head=self.heads,
        )
        keys = rearrange(
            self.key_projection(tokens),
            "batch time (head channel) -> batch time head channel",
            head=self.heads,
        )
        values = rearrange(
            self.value_projection(tokens),
            "batch time (head channel) -> batch time head channel",
            head=self.heads,
        )

        # lazy bum doesnt use his own normalization fns
        queries = F.normalize(F.silu(queries), dim=-1)
        keys = F.normalize(F.silu(keys), dim=-1)
        values = F.silu(values)

        decay_logits = rearrange(
            self.decay_projection(tokens),
            "batch time (head channel) -> batch time head channel",
            head=self.heads,
        ).float()
        log_decay = self.minimum_log_decay * torch.sigmoid(
            torch.exp(self.log_decay_scale)[None, None, :, None]
            * (decay_logits + self.decay_bias[None, None])
        )
        write_strength = torch.sigmoid(self.write_projection(tokens).float())
        output, state = chunk_kimi_delta_attention(
            queries,
            keys,
            values,
            log_decay,
            write_strength,
            initial_state,
            self.chunk_size,
        )

        output_gate = torch.sigmoid(
            rearrange(
                self.output_gate_projection(tokens),
                "batch time (head channel) -> batch time head channel",
                head=self.heads,
            )
        )
        output = self.output_norm(output) * output_gate
        output = self.output_projection(
            rearrange(output, "batch time head channel -> batch time (head channel)")
        )
        return output, state
