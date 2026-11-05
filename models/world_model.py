import torch
from beartype import beartype
from diffusers.models.embeddings import GaussianFourierProjection
from einops import rearrange, repeat
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn
from torch.distributed import ProcessGroup
from torch.nn.attention.flex_attention import BlockMask, create_block_mask
from torch.utils.checkpoint import checkpoint
from models.ddt import AudioVisualDDTHead
from models.model_types import AudioTokenConfig
from models.sequence_parallel import shard_sequence
from models.tokens import AudioVisualTokenProjector
from models.transformers import (
    AdaptiveAttentionBlock,
    sinusoidal_timestep_embedding,
)


class AudioVisualFlowPredictor(nn.Module):
    def __init__(
        self,
        video_dimension: int,
        audio_dimension: int,
        action_dimension: int,
        backbone_hidden_dimension: int,
        backbone_heads: int,
        backbone_layers: int,
        diffusion_head_hidden_dimension: int,
        diffusion_head_heads: int,
        diffusion_head_layers: int,
        mlp_ratio: float,
        dropout: float,
        noise_time_embedding: str = "sinusoidal",
        spatial_grid: tuple[int, int] | None = None,
        audio_tokens: AudioTokenConfig | None = None,
        prediction: str = "velocity",
        backbone_key_value_heads: int | None = None,
        diffusion_head_key_value_heads: int | None = None,
        attention_window: int | None = None,
        attention_backend: str = "sdpa",
        context_noise_conditioning: bool = False,
    ) -> None:
        super().__init__()
        self.activation_checkpointing = False
        self.sequence_group: ProcessGroup | None = None
        self.backbone_hidden_dimension = backbone_hidden_dimension
        self.audio_tokens = audio_tokens or AudioTokenConfig()
        if noise_time_embedding not in {"sinusoidal", "gaussian_fourier"}:
            raise ValueError("Unknown noise time embedding")
        self.noise_time_embedding = noise_time_embedding
        if prediction not in {"velocity", "data"}:
            raise ValueError("Unknown flow prediction target")
        self.prediction = prediction
        self.attention_window = attention_window
        self.attention_backend = attention_backend
        self.noise_time_projection = (
            GaussianFourierProjection(
                backbone_hidden_dimension // 2,
                scale=1.0,
                set_W_to_weight=False,
                log=False,
            )
            if noise_time_embedding == "gaussian_fourier"
            else nn.Identity()
        )
        self.null_action = nn.Parameter(torch.zeros(action_dimension))
        self.backbone_target_tokens = AudioVisualTokenProjector(
            video_dimension,
            audio_dimension,
            backbone_hidden_dimension,
            spatial_grid,
            self.audio_tokens,
        )
        self.backbone_context_tokens = AudioVisualTokenProjector(
            video_dimension,
            audio_dimension,
            backbone_hidden_dimension,
            spatial_grid,
            self.audio_tokens,
        )
        self.backbone_action_projection = nn.Linear(
            action_dimension,
            backbone_hidden_dimension,
            bias=False,
        )
        self.start_time_embedding = nn.Sequential(
            nn.Linear(backbone_hidden_dimension, backbone_hidden_dimension * 4),
            nn.SiLU(),
            nn.Linear(backbone_hidden_dimension * 4, backbone_hidden_dimension),
        )
        self.duration_embedding = nn.Sequential(
            nn.Linear(backbone_hidden_dimension, backbone_hidden_dimension * 4),
            nn.SiLU(),
            nn.Linear(backbone_hidden_dimension * 4, backbone_hidden_dimension),
        )
        self.context_noise_embedding = (
            nn.Sequential(
                nn.Linear(backbone_hidden_dimension, backbone_hidden_dimension * 4),
                nn.SiLU(),
                nn.Linear(backbone_hidden_dimension * 4, backbone_hidden_dimension),
            )
            if context_noise_conditioning
            else None
        )
        self.backbone_blocks = nn.ModuleList(
            AdaptiveAttentionBlock(
                backbone_hidden_dimension,
                backbone_heads,
                mlp_ratio,
                dropout,
                backbone_key_value_heads,
                attention_backend,
            )
            for layer_index in range(backbone_layers)
        )
        self.diffusion_head = AudioVisualDDTHead(
            video_dimension,
            audio_dimension,
            action_dimension,
            backbone_hidden_dimension,
            diffusion_head_hidden_dimension,
            diffusion_head_heads,
            diffusion_head_layers,
            mlp_ratio,
            dropout,
            spatial_grid,
            self.audio_tokens,
            diffusion_head_key_value_heads,
            attention_backend,
        )

    @jaxtyped(typechecker=beartype)
    def noise_time_features(self, time: Float[Tensor, "batch"]) -> Float[Tensor, "batch hidden"]:  # noqa: F821
        if self.noise_time_embedding == "gaussian_fourier":
            return self.noise_time_projection(time)
        return sinusoidal_timestep_embedding(time, self.backbone_hidden_dimension)

    @jaxtyped(typechecker=beartype)
    def condition_actions(
        self,
        video: Float[Tensor, "batch target_time patch video_feature"],
        actions: Float[Tensor, "batch target_time action"] | None,
    ) -> Float[Tensor, "batch target_time action"]:
        expected_shape = (*video.shape[:2], self.null_action.shape[0])
        if actions is not None:
            if actions.shape != expected_shape:
                raise ValueError(f"Expected action conditions shaped {torch.Size(expected_shape)}")
            return actions
        return repeat(
            self.null_action,
            "action -> batch time action",
            batch=video.shape[0],
            time=video.shape[1],
        )

    def frame_attention_mask(self, target_steps: int, tokens_per_step: int, device: torch.device) -> Tensor | BlockMask:
        frame_indices = repeat(torch.arange(target_steps, device=device), "time -> (time token)", token=tokens_per_step)
        window = self.attention_window or target_steps

        def frame_causal(batch_index: Tensor, head_index: Tensor, query_index: Tensor, key_index: Tensor) -> Tensor:
            frame_gap = frame_indices[query_index] - frame_indices[key_index]
            return (frame_gap >= 0) & (frame_gap < window)

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
        condition = self.start_time_embedding(self.noise_time_features(start_time))
        condition = condition + self.duration_embedding(self.noise_time_features(end_time - start_time))
        if self.context_noise_embedding is None:
            return condition
        level = torch.zeros_like(start_time) if context_noise is None else context_noise
        return condition + self.context_noise_embedding(self.noise_time_features(level))

    def backbone_targets(
        self,
        noisy_video: Tensor,
        noisy_audio: Tensor,
        context_video: Tensor,
        context_audio: Tensor,
        target_actions: Tensor,
        context_actions: Tensor | None,
        condition: Tensor,
        attention_mask: Tensor | BlockMask,
    ) -> Tensor:
        targets = self.backbone_target_tokens(
            noisy_video, noisy_audio, 0, self.backbone_action_projection(target_actions)
        )
        if self.sequence_group is not None:
            targets = shard_sequence(targets, self.sequence_group)
        context = self.backbone_context_tokens(
            context_video,
            context_audio,
            -context_video.shape[1],
            None if context_actions is None else self.backbone_action_projection(context_actions),
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
        actions: Float[Tensor, "batch target_time action"] | None,
        context_actions: Float[Tensor, "batch context_time action"] | None = None,
        context_noise: Float[Tensor, "batch"] | None = None,  # noqa: F821
    ) -> tuple[
        Float[Tensor, "batch target_time patch video_feature"],
        Float[Tensor, "batch target_time audio_feature"],
    ]:
        if noisy_video.shape[:2] != noisy_audio.shape[:2]:
            raise ValueError("Target audio and video times must align")
        if context_video.shape[:2] != context_audio.shape[:2]:
            raise ValueError("Context audio and video times must align")
        if torch.any(end_time < start_time):
            raise ValueError("Flow end time must not precede start time")
        if context_actions is not None and context_actions.shape[2] != self.null_action.shape[0]:
            raise ValueError("Previous executed actions must align with observation history")
        target_actions = self.condition_actions(noisy_video, actions)
        attention_mask = self.frame_attention_mask(
            noisy_video.shape[1], noisy_video.shape[2] + self.audio_tokens.tokens_per_step, noisy_video.device
        )
        condition = self.flow_condition(start_time, end_time, context_noise)
        targets = self.backbone_targets(
            noisy_video,
            noisy_audio,
            context_video,
            context_audio,
            target_actions,
            context_actions,
            condition,
            attention_mask,
        )
        video_output, audio_output = self.diffusion_head(
            noisy_video,
            noisy_audio,
            context_video,
            context_audio,
            target_actions,
            targets,
            condition,
            attention_mask,
        )
        if self.prediction == "velocity":
            return video_output, audio_output
        remaining = (1.0 - start_time).clamp_min(0.05)
        return (
            (video_output - noisy_video) / rearrange(remaining, "batch -> batch 1 1 1"),
            (audio_output - noisy_audio) / rearrange(remaining, "batch -> batch 1 1"),
        )
