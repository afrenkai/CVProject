import torch
from beartype import beartype
from diffusers.models.embeddings import GaussianFourierProjection
from einops import rearrange, repeat
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn

from architecture.pretraining.ddt import AudioVisualDDTHead
from architecture.pretraining.history_mixer import AudioVisualHistoryMixer
from architecture.pretraining.tokens import AudioVisualTokenProjector
from architecture.pretraining.transformers import (
    AdaptiveAttentionBlock,
    sinusoidal_timestep_embedding,
)
from architecture.pretraining.types import HybridHistoryConfig


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
        backbone_attention: str = "quadratic",
        hybrid_history: HybridHistoryConfig | None = None,
        noise_time_embedding: str = "sinusoidal",
        spatial_grid: tuple[int, int] | None = None,
    ) -> None:
        super().__init__()
        self.backbone_hidden_dimension = backbone_hidden_dimension
        if noise_time_embedding not in {"sinusoidal", "gaussian_fourier"}:
            raise ValueError("Unknown noise time embedding")
        self.noise_time_embedding = noise_time_embedding
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
        )
        self.backbone_context_tokens = AudioVisualTokenProjector(
            video_dimension,
            audio_dimension,
            backbone_hidden_dimension,
            spatial_grid,
        )
        self.backbone_action_projection = nn.Linear(
            action_dimension,
            backbone_hidden_dimension,
            bias=False,
        )
        self.outcome_head = nn.Sequential(
            nn.RMSNorm(backbone_hidden_dimension),
            nn.Linear(backbone_hidden_dimension, backbone_hidden_dimension * 2),
            nn.SiLU(),
            nn.Linear(backbone_hidden_dimension * 2, 2),
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
        if backbone_attention not in {"quadratic", "linear", "hybrid_kda"}:
            raise ValueError("Unknown backbone attention")
        self.backbone_attention = backbone_attention
        self.backbone_blocks = nn.ModuleList(
            AdaptiveAttentionBlock(
                backbone_hidden_dimension,
                backbone_heads,
                mlp_ratio,
                dropout,
            )
            for layer_index in range(backbone_layers if backbone_attention != "hybrid_kda" else 0)
        )
        if backbone_attention == "hybrid_kda" and hybrid_history is None:
            raise ValueError("Hybrid KDA history requires its block settings")
        self.linear_mixer = AudioVisualHistoryMixer(
            video_dimension,
            audio_dimension,
            action_dimension,
            backbone_hidden_dimension,
            backbone_heads,
            backbone_layers,
            mlp_ratio,
            dropout,
            hybrid_history if backbone_attention == "hybrid_kda" else None,
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
        )

    @jaxtyped(typechecker=beartype)
    def noise_time_features(self, time: Float[Tensor, "batch"]) -> Float[Tensor, "batch hidden"]:  # noqa: F821
        if self.noise_time_embedding == "gaussian_fourier":
            return self.noise_time_projection(time)
        return sinusoidal_timestep_embedding(time, self.backbone_hidden_dimension)

    @jaxtyped(typechecker=beartype)
    def predict_transition_outcomes(
        self,
        context_video: Float[Tensor, "batch context_time patch video_feature"],
        context_audio: Float[Tensor, "batch context_time audio_feature"],
        actions: Float[Tensor, "batch target_time action"],
        context_actions: Float[Tensor, "batch context_time action"] | None = None,
    ) -> tuple[
        Float[Tensor, "batch target_time"],
        Float[Tensor, "batch target_time"],
    ]:
        if context_video.shape[:2] != context_audio.shape[:2]:
            raise ValueError("Context audio and video times must align")
        if self.backbone_attention in {"linear", "hybrid_kda"}:
            pooled_context = self.linear_mixer.encode_context(context_video, context_audio, context_actions)
        else:
            context_steps = context_video.shape[1]
            context_tokens = self.backbone_context_tokens(context_video, context_audio, -context_steps)
            pooled_context = rearrange(
                context_tokens,
                "batch (time token) hidden -> batch time token hidden",
                time=context_steps,
            )[:, -1].mean(dim=1)
        outcome_tokens = rearrange(pooled_context, "batch hidden -> batch 1 hidden") + self.backbone_action_projection(
            actions
        )
        outcomes = self.outcome_head(outcome_tokens)
        return outcomes[:, :, 0], outcomes[:, :, 1]

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

        batch_size, target_steps, video_patches = noisy_video.shape[:3]
        context_steps = context_video.shape[1]
        target_actions = self.condition_actions(noisy_video, actions)
        if context_actions is not None and context_actions.shape != (
            batch_size,
            context_steps,
            self.null_action.shape[0],
        ):
            raise ValueError("Previous executed actions must align with observation history")
        tokens_per_step = video_patches + 1
        frame_indices = repeat(
            torch.arange(target_steps, device=noisy_video.device),
            "time -> (time token)",
            token=tokens_per_step,
        )
        attention_mask = frame_indices[None, :] <= frame_indices[:, None]
        condition = self.start_time_embedding(self.noise_time_features(start_time))
        condition = condition + self.duration_embedding(self.noise_time_features(end_time - start_time))
        if self.backbone_attention in {"linear", "hybrid_kda"}:
            mixed_targets = self.linear_mixer(
                context_video,
                context_audio,
                noisy_video,
                noisy_audio,
                target_actions,
                context_actions,
            )[0]
            mixed_targets = mixed_targets + rearrange(condition, "batch hidden -> batch 1 hidden")
            targets = repeat(
                mixed_targets,
                "batch time hidden -> batch (time token) hidden",
                token=tokens_per_step,
            )
        else:
            targets = self.backbone_target_tokens(
                noisy_video,
                noisy_audio,
                0,
                self.backbone_action_projection(target_actions),
            )
            context = self.backbone_context_tokens(
                context_video,
                context_audio,
                -context_steps,
                None if context_actions is None else self.backbone_action_projection(context_actions),
            )
            global_condition = rearrange(condition, "batch hidden -> batch 1 hidden")
            for block in self.backbone_blocks:
                targets = block(targets, context, global_condition, attention_mask)

        return self.diffusion_head(
            noisy_video,
            noisy_audio,
            context_video,
            context_audio,
            target_actions,
            targets,
            condition,
            attention_mask,
        )
