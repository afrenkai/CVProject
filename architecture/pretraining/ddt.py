import torch
from einops import rearrange
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from architecture.pretraining.model_types import AudioTokenConfig
from architecture.pretraining.tokens import AudioVisualTokenProjector
from architecture.pretraining.transformers import AdaptiveAttentionBlock


class AdaptiveOutputProjection(nn.Module):
    def __init__(self, hidden_dimension: int, output_dimension: int) -> None:
        """
        stole from Boyang and changed to look a little clener.
        modulation is silu -> affine_fn for 2x h_dim
        zero init w,b for all since initially want to lean on the frozen representations

        """
        super().__init__()
        self.normalization = nn.RMSNorm(hidden_dimension, elementwise_affine=False)
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dimension, hidden_dimension * 2),
        )
        self.output_projection = nn.Linear(hidden_dimension, output_dimension)
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, tokens: Tensor, condition: Tensor) -> Tensor:
        shift, scale = self.modulation(condition).chunk(2, dim=-1)
        return self.output_projection(self.normalization(tokens) * (1 + scale) + shift)


class AudioVisualDDTHead(nn.Module):
    def __init__(
        self,
        video_dimension: int,
        audio_dimension: int,
        action_dimension: int,
        backbone_dimension: int,
        hidden_dimension: int,
        heads: int,
        layers: int,
        mlp_ratio: float,
        dropout: float,
        spatial_grid: tuple[int, int] | None = None,
        audio_tokens: AudioTokenConfig | None = None,
        key_value_heads: int | None = None,
    ) -> None:
        super().__init__()
        self.activation_checkpointing = False
        self.audio_tokens = audio_tokens or AudioTokenConfig()
        self.target_tokens = AudioVisualTokenProjector(
            video_dimension,
            audio_dimension,
            hidden_dimension,
            spatial_grid,
            self.audio_tokens,
        )
        self.context_tokens = AudioVisualTokenProjector(
            video_dimension,
            audio_dimension,
            hidden_dimension,
            spatial_grid,
            self.audio_tokens,
        )
        self.action_projection = nn.Linear(
            action_dimension,
            hidden_dimension,
            bias=False,
        )
        self.condition_projection = nn.Linear(backbone_dimension, hidden_dimension)
        self.blocks = nn.ModuleList(
            AdaptiveAttentionBlock(hidden_dimension, heads, mlp_ratio, dropout, key_value_heads)
            for layer_index in range(layers)
        )
        self.video_output = AdaptiveOutputProjection(
            hidden_dimension,
            video_dimension,
        )
        self.audio_output = AdaptiveOutputProjection(
            hidden_dimension,
            audio_dimension // self.audio_tokens.tokens_per_step,
        )

    def forward(
        self,
        noisy_video: Tensor,
        noisy_audio: Tensor,
        context_video: Tensor,
        context_audio: Tensor,
        target_actions: Tensor,
        backbone_targets: Tensor,
        backbone_condition: Tensor,
        attention_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        target_steps = noisy_video.shape[1]
        video_patches = noisy_video.shape[2]
        targets = self.target_tokens(
            noisy_video,
            noisy_audio,
            0,
            self.action_projection(target_actions),
        )
        context = self.context_tokens(context_video, context_audio, -context_video.shape[1])
        condition = self.condition_projection(
            F.silu(backbone_targets + rearrange(backbone_condition, "batch hidden -> batch 1 hidden"))
        )
        for block in self.blocks:
            if self.activation_checkpointing and torch.is_grad_enabled():
                targets = checkpoint(block, targets, context, condition, attention_mask, use_reentrant=False)
            else:
                targets = block(targets, context, condition, attention_mask)

        tokens_per_step = video_patches + self.audio_tokens.tokens_per_step
        predictions = rearrange(
            targets,
            "batch (time token) hidden -> batch time token hidden",
            time=target_steps,
            token=tokens_per_step,
        )
        output_condition = rearrange(
            condition,
            "batch (time token) hidden -> batch time token hidden",
            time=target_steps,
            token=tokens_per_step,
        )
        video_velocity = self.video_output(
            predictions[:, :, :video_patches],
            output_condition[:, :, :video_patches],
        )
        audio_velocity = self.audio_output(
            predictions[:, :, video_patches:],
            output_condition[:, :, video_patches:],
        )
        return video_velocity, rearrange(audio_velocity, "batch time token feature -> batch time (token feature)")
