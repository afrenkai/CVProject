import math
from collections.abc import Sequence
import torch
from beartype import beartype
from einops import rearrange
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn
from torch.nn import functional
from architecture.pretraining.kda import KimiDeltaAttention


class CausalLocalAttention(nn.Module):
    """
    causal swiAttn. still (qk^{T}/sqrt(d_h))V but windowed out using a logical and from torch.
    takes ideas from https://arxiv.org/html/2505.02074v1
    @param hidden_dimension int for the latent space to occupy
    @param heads: attn heads
    @param window_steps


    """
    def __init__(self, hidden_dimension: int, heads: int, window_steps: int) -> None:
        super().__init__()
        if hidden_dimension % heads or window_steps <= 0:
            raise ValueError(
                "Attention requires divisible dimensions and a positive window"
            )
        self.heads = heads
        self.window_steps = window_steps
        self.query_projection = nn.Linear(
            hidden_dimension, hidden_dimension, bias=False
        )
        self.key_projection = nn.Linear(hidden_dimension, hidden_dimension, bias=False)
        self.value_projection = nn.Linear(
            hidden_dimension, hidden_dimension, bias=False
        )
        self.output_projection = nn.Linear(
            hidden_dimension, hidden_dimension, bias=False
        )

    @jaxtyped(typechecker=beartype)
    def forward(
        self, tokens: Float[Tensor, "batch time hidden"]
    ) -> Float[Tensor, "batch time hidden"]:
        queries = rearrange(
            self.query_projection(tokens),
            "batch time (head channel) -> batch head time channel",
            head=self.heads,
        )
        keys = rearrange(
            self.key_projection(tokens),
            "batch time (head channel) -> batch head time channel",
            head=self.heads,
        )
        values = rearrange(
            self.value_projection(tokens),
            "batch time (head channel) -> batch head time channel",
            head=self.heads,
        )
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        distance = positions[:, None] - positions[None, :]
        attention_mask = torch.logical_and(
            distance >= 0,
            distance < self.window_steps,
        )
        attended = functional.scaled_dot_product_attention(
            queries,
            keys,
            values,
            attn_mask=attention_mask,
        )
        return self.output_projection(
            rearrange(
                attended,
                "batch head time channel -> batch time (head channel)",
            )
        )


class HybridAudioEncoder(nn.Module):
    def __init__(
        self,
        input_channels: int,
        hidden_dimension: int,
        latent_dimension: int,
        downsample_strides: Sequence[int],
        block_pattern: Sequence[str],
        kda_heads: int,
        kda_head_dimension: int,
        kda_chunk_size: int,
        kda_minimum_log_decay: float,
        attention_heads: int,
        attention_window_steps: int,
    ) -> None:
        """
        setup for waveform -> latent. best way is linear/softmax attn with some ratio
        alternates between kimi delta and transformer layers
        operates on channel_dim
        ratio can be passed as seq but maybe better to parse as a frac non-natively?

        @param input_channels: int how many channels which are for mlp/transformer in
        @param h_dim: int dim of conv output for downsampling waveform
        @param latent_dim: int what dim/size of emb space 
        @param downsample_strides: seq of int representing strides of conv for each. need to do a len check
        @param block_pattern: seq(str) like "K","K","K","A","K","K","K","A" for the pattern of blocks of kda vs attn
        @param kda_heads: num_heads for kda
        @param kda_head_dimension: 
        @param kda_chunk_size: size of chunk to perform Linear ops on for KDA
        @param kda_minimum_log_decay: smallest amount to log-decay the logits of kda
        @param attention_heads: how many heads to use in softmax attn
        @param attn_window_steps: since we are using causal attn (still softmax attn, but a little faster than naive SDPA) how big do we want the sliding window steps to be?
        """
        super().__init__()
        if not downsample_strides or any(stride <= 0 for stride in downsample_strides):
            raise ValueError("Audio downsample strides must be positive")

        stem_layers: list[nn.Module] = []
        stem_input_channels = input_channels
        for stride in downsample_strides:
            stem_layers.extend(
                (
                    nn.Conv1d(
                        stem_input_channels,
                        hidden_dimension,
                        kernel_size=stride,
                        stride=stride,
                    ),
                    nn.GELU(),
                )
            )
            stem_input_channels = hidden_dimension
        self.stem = nn.Sequential(*stem_layers)
        self.total_stride = math.prod(downsample_strides)

        # whole concept of mixers comes I think from Mamba?
        # https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/mixer_seq_simple.py
        mixers: list[nn.Module] = []
        for block_kind in block_pattern:
            if block_kind == "kda":
                mixers.append(
                    KimiDeltaAttention(
                        hidden_dimension,
                        kda_heads,
                        kda_head_dimension,
                        kda_chunk_size,
                        kda_minimum_log_decay,
                    )
                )
            elif block_kind == "attention":
                mixers.append(
                    CausalLocalAttention(
                        hidden_dimension,
                        attention_heads,
                        attention_window_steps,
                    )
                )
            else:
                raise ValueError(f"Unknown audio mixer: {block_kind}")
        self.mixers = nn.ModuleList(mixers)
        self.mixer_norms = nn.ModuleList(
            nn.RMSNorm(hidden_dimension) for block_kind in block_pattern
        )
        self.feed_forward_norms = nn.ModuleList(
            nn.RMSNorm(hidden_dimension) for block_kind in block_pattern
        )
        self.feed_forwards = nn.ModuleList(
            nn.Sequential(
                nn.Linear(hidden_dimension, hidden_dimension * 4),
                nn.SiLU(),
                nn.Linear(hidden_dimension * 4, hidden_dimension),
            )
            for block_kind in block_pattern
        )
        self.output_projection = nn.Linear(hidden_dimension, latent_dimension)
        self.kda_block_count = block_pattern.count("kda")

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        waveform: Float[Tensor, "batch channel sample"],
        initial_states: list[Tensor] | None = None,
    ) -> tuple[Float[Tensor, "batch time latent"], list[Tensor]]:
        """
        takes in waveform (tensor) with some initial states and runs fwd pass. 
        @param waveform: float tensor of shape bxcxsample (or audio_sample, depending on which file you look at)
        @param initial_states: list of tensors for initializing the kda state matrix. can be optional. 
        
        Returns:
            latent representation which is 3 tensor of b x t x l
            l is inherited from the model class, same as t
            also returns the state matrix for next states
        """
        if waveform.shape[-1] % self.total_stride:
            raise ValueError(
                f"Audio samples must be divisible by encoder stride {self.total_stride}"
            )
        if initial_states is not None and len(initial_states) != self.kda_block_count:
            raise ValueError(f"Expected {self.kda_block_count} KDA states")

        tokens = rearrange(
            self.stem(waveform), "batch hidden time -> batch time hidden"
        )
        next_states: list[Tensor] = []
        # I am not sure if there is a "smarter" way to do iteration through this than a counter and enumerate. maybe something with zip?
        # for now, counter = 0, counter+=1 if valid idx else throw err
        state_index = 0
        for mixer_index, mixer in enumerate(self.mixers):
            normalized = self.mixer_norms[mixer_index](tokens)
            if isinstance(mixer, KimiDeltaAttention):
                initial_state = (
                    None if initial_states is None else initial_states[state_index]
                )
                mixed, next_state = mixer(normalized, initial_state)
                next_states.append(next_state)
                state_index += 1
            elif isinstance(mixer, CausalLocalAttention):
                mixed = mixer(normalized)
            else:
                raise TypeError(f"Unsupported mixer type: {type(mixer).__name__}")
            tokens = tokens + mixed
            tokens = tokens + self.feed_forwards[mixer_index](
                self.feed_forward_norms[mixer_index](tokens)
            )
        return self.output_projection(tokens), next_states
