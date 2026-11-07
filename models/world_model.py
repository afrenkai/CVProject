import torch
from beartype import beartype
from diffusers.models.embeddings import GaussianFourierProjection
from einops import rearrange, repeat
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn
from torch.nn.attention.flex_attention import BlockMask, create_block_mask
from torch.utils.checkpoint import checkpoint
from models.ddt import AudioVisualDDTHead
from models.model_types import AudioTokenConfig, WorldModelConfig
from models.tokens import AudioVisualTokenProjector
from models.transformers import AdaptiveAttentionBlock


def time_embedding(h_dim: int) -> nn.Sequential:
    """
    T emb for dit,
    need to make latent_dim arg and pass both by cfg

    @param h_dim: int = hidden dim

    @param emb_dim: int = embedding dim default 4x h_dim

    Returns:
        Timestep Embedding (Affine Fn (h,e), sigma, Affine Fn(e, h))
    """
    #TODO: refactor all callers to pass emb_dim
    return nn.Sequential(
        nn.Linear(h_dim, h_dim * 4),
        nn.SiLU(),
        nn.Linear(h_dim * 4, h_dim)
    )


class AudioVisualFlowPredictor(nn.Module):
    def __init__(
        self, video_dimension: int, audio_dimension: int, audio_tokens: AudioTokenConfig, config: WorldModelConfig
    ) -> None:
        super().__init__()
        hidden = config.backbone.hidden_dimension
        self.activation_checkpointing = False
        self.audio_tokens = audio_tokens
        self.attention_backend = config.attention_backend
        self.noise_time_projection = GaussianFourierProjection(hidden // 2, scale=1.0, set_W_to_weight=False, log=False)
        self.null_action = nn.Parameter(torch.zeros(config.action_dimension))
        self.backbone_target_tokens = AudioVisualTokenProjector(
            video_dimension, audio_dimension, hidden, config.spatial_grid, audio_tokens
        )
        self.backbone_context_tokens = AudioVisualTokenProjector(
            video_dimension, audio_dimension, hidden, config.spatial_grid, audio_tokens
        )
        self.backbone_action_projection = nn.Linear(config.action_dimension, hidden, bias=False)
        self.start_time_embedding = time_embedding(hidden)
        self.duration_embedding = time_embedding(hidden)
        self.context_noise_embedding = time_embedding(hidden)
        self.backbone_blocks = nn.ModuleList(
            AdaptiveAttentionBlock(hidden, config.backbone.heads, config.mlp_ratio, config.attention_backend)
            for layer_index in range(config.backbone.layers)
        )
        self.diffusion_head = AudioVisualDDTHead(video_dimension, audio_dimension, audio_tokens, config)

    def frame_attention_mask(self, target_steps: int, tokens_per_step: int, device: torch.device) -> Tensor | BlockMask:
        frame_indices = repeat(torch.arange(target_steps, device=device), "time -> (time token)", token=tokens_per_step)

        def frame_causal(batch_index: Tensor, head_index: Tensor, query_index: Tensor, key_index: Tensor) -> Tensor:
            return frame_indices[query_index] >= frame_indices[key_index]

        length = frame_indices.shape[0]
        if self.attention_backend == "sdpa":
            sequence = torch.arange(length, device=device)
            return frame_causal(sequence, sequence, sequence[:, None], sequence[None, :])
        return create_block_mask(frame_causal, None, None, length, length, device=device)

    @jaxtyped(typechecker=beartype)
    def flow_condition(
        self,
        start_time: Float[Tensor, "batch"],  # noqa: F821
        end_time: Float[Tensor, "batch"],  # noqa: F821
        context_noise: Float[Tensor, "batch"] | None,  # noqa: F821
    ) -> Float[Tensor, "batch hidden"]:
        level = torch.zeros_like(start_time) if context_noise is None else context_noise
        return (
            self.start_time_embedding(self.noise_time_projection(start_time))
            + self.duration_embedding(self.noise_time_projection(end_time - start_time))
            + self.context_noise_embedding(self.noise_time_projection(level))
        )

    @jaxtyped(typechecker=beartype)
    def backbone_targets(
        self,
        noisy_video: Float[Tensor, "batch target_time patch video_feature"],
        noisy_audio: Float[Tensor, "batch target_time audio_feature"],
        context_video: Float[Tensor, "batch context_time patch video_feature"],
        context_audio: Float[Tensor, "batch context_time audio_feature"],
        target_actions: Float[Tensor, "batch target_time action"],
        context_actions: Float[Tensor, "batch context_time action"],
        condition: Float[Tensor, "batch hidden"],
        attention_mask: Tensor | BlockMask,
    ) -> Float[Tensor, "batch token hidden"]:
        targets = self.backbone_target_tokens(
            noisy_video, noisy_audio, 0, self.backbone_action_projection(target_actions)
        )
        context = self.backbone_context_tokens(
            context_video, context_audio, -context_video.shape[1], self.backbone_action_projection(context_actions)
        )
        global_condition = rearrange(condition, "batch hidden -> batch 1 hidden")
        for block in self.backbone_blocks:
            if self.activation_checkpointing and torch.is_grad_enabled():
                targets = checkpoint(block, targets, context, global_condition, attention_mask, use_reentrant=False)
            else:
                targets = block(targets, context, global_condition, attention_mask)
        return targets

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        noisy_video: Float[Tensor, "batch target_time patch video_feature"],
        noisy_audio: Float[Tensor, "batch target_time audio_feature"],
        start_time: Float[Tensor, "batch"],  # noqa: F821
        end_time: Float[Tensor, "batch"],  # noqa: F821
        context_video: Float[Tensor, "batch context_time patch video_feature"],
        context_audio: Float[Tensor, "batch context_time audio_feature"],
        actions: Float[Tensor, "batch target_time action"],
        context_actions: Float[Tensor, "batch context_time action"],
        context_noise: Float[Tensor, "batch"] | None = None,  # noqa: F821
    ) -> tuple[
        Float[Tensor, "batch target_time patch video_feature"],
        Float[Tensor, "batch target_time audio_feature"],
    ]:
        assert torch.all(end_time >= start_time)
        attention_mask = self.frame_attention_mask(
            noisy_video.shape[1], noisy_video.shape[2] + self.audio_tokens.tokens_per_step, noisy_video.device
        )
        condition = self.flow_condition(start_time, end_time, context_noise)
        targets = self.backbone_targets(
            noisy_video, noisy_audio, context_video, context_audio, actions, context_actions, condition, attention_mask
        )
        video_output, audio_output = self.diffusion_head(
            noisy_video, noisy_audio, context_video, context_audio, actions, targets, condition, attention_mask
        )
        remaining = (1.0 - start_time).clamp_min(0.05)
        return (
            (video_output - noisy_video) / rearrange(remaining, "batch -> batch 1 1 1"),
            (audio_output - noisy_audio) / rearrange(remaining, "batch -> batch 1 1"),
        )
