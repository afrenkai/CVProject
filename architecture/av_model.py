import torch
from beartype import beartype
from einops import rearrange, repeat
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn
from architecture.pretraining.ddt import AudioVisualDDTHead
from architecture.pretraining.tokens import AudioVisualTokenProjector
from architecture.pretraining.transformers import (
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
    ) -> None:
        super().__init__()
        self.backbone_hidden_dimension = backbone_hidden_dimension
        self.null_action = nn.Parameter(torch.zeros(action_dimension))
        self.backbone_target_tokens = AudioVisualTokenProjector(
            video_dimension,
            audio_dimension,
            backbone_hidden_dimension,
        )
        self.backbone_context_tokens = AudioVisualTokenProjector(
            video_dimension,
            audio_dimension,
            backbone_hidden_dimension,
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
        self.backbone_blocks = nn.ModuleList(
            AdaptiveAttentionBlock(
                backbone_hidden_dimension,
                backbone_heads,
                mlp_ratio,
                dropout,
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
        )

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        noisy_video: Float[Tensor, "batch target_time patch video_feature"],
        noisy_audio: Float[Tensor, "batch target_time audio_feature"],
        start_time: Float[Tensor, "batch"],
        end_time: Float[Tensor, "batch"],
        context_video: Float[Tensor, "batch context_time patch video_feature"],
        context_audio: Float[Tensor, "batch context_time audio_feature"],
        actions: Float[Tensor, "batch target_time action"] | None,
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
        target_actions = repeat(
            self.null_action,
            "action -> batch time action",
            batch=batch_size,
            time=target_steps,
        )
        if actions is not None:
            if actions.shape != target_actions.shape:
                raise ValueError(
                    f"Expected action conditions shaped {target_actions.shape}"
                )
            target_actions = actions
        target_action_embeddings = self.backbone_action_projection(target_actions)
        context_action_embeddings = context_video.new_zeros(
            (batch_size, context_steps, self.backbone_hidden_dimension)
        )

        target_positions = sinusoidal_timestep_embedding(
            torch.arange(
                target_steps,
                device=noisy_video.device,
                dtype=noisy_video.dtype,
            ),
            self.backbone_hidden_dimension,
        )
        context_positions = sinusoidal_timestep_embedding(
            torch.arange(
                -context_steps,
                0,
                device=noisy_video.device,
                dtype=noisy_video.dtype,
            ),
            self.backbone_hidden_dimension,
        )
        targets = self.backbone_target_tokens(
            noisy_video,
            noisy_audio,
            target_positions,
            target_action_embeddings,
        )
        context = self.backbone_context_tokens(
            context_video,
            context_audio,
            context_positions,
            context_action_embeddings,
        )

        tokens_per_step = video_patches + 1
        frame_indices = repeat(
            torch.arange(target_steps, device=noisy_video.device),
            "time -> (time token)",
            token=tokens_per_step,
        )
        attention_mask = frame_indices[None, :] <= frame_indices[:, None]
        condition = self.start_time_embedding(
            sinusoidal_timestep_embedding(start_time, self.backbone_hidden_dimension)
        )
        condition = condition + self.duration_embedding(
            sinusoidal_timestep_embedding(
                end_time - start_time,
                self.backbone_hidden_dimension,
            )
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
