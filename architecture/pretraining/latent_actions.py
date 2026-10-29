import torch
from beartype import beartype
from einops import einsum, reduce, repeat
from jaxtyping import Float, Int, jaxtyped
from torch import Tensor, nn
from torch.nn import functional as F
from architecture.pretraining.types import ActionAudioConditioning


class LatentActionModel(nn.Module):
    def __init__(
        self,
        video_dimension: int,
        audio_dimension: int,
        hidden_dimension: int,
        action_dimension: int,
        code_count: int,
        ema_decay: float,
        ema_epsilon: float,
        audio_conditioning: ActionAudioConditioning = "transition",
    ) -> None:
        """
        VQ-VAE for operating in a set action dim.
        insp from Jack Lu fro m Mengye's lab who did this for xHC

        """
        super().__init__()
        if code_count < 2 or not 0.0 <= ema_decay < 1.0 or ema_epsilon <= 0.0:
            raise ValueError("Latent-action codebook settings are invalid")
        transition_dimension = 2 * (video_dimension + audio_dimension)
        self.encoder = nn.Sequential(
            nn.Linear(transition_dimension, hidden_dimension),
            nn.SiLU(),
            nn.Linear(hidden_dimension, action_dimension),
        )
        codebook = F.normalize(torch.randn(code_count, action_dimension), dim=-1)
        self.register_buffer("codebook", codebook)
        self.register_buffer("ema_counts", torch.ones(code_count))
        self.register_buffer("ema_sums", codebook.clone())
        self.code_count = code_count
        self.ema_decay = ema_decay
        self.ema_epsilon = ema_epsilon
        self.audio_conditioning = audio_conditioning

    @jaxtyped(typechecker=beartype)
    def sequence_actions(
        self,
        video: Float[Tensor, "batch time patch video_feature"],
        audio: Float[Tensor, "batch time audio_feature"],
        context_steps: int,
    ) -> tuple[
        Float[Tensor, "batch target_time action"],
        Int[Tensor, "batch target_time"],
        Float[Tensor, "batch target_time action"],
        Float[Tensor, ""],
    ]:
        """mask future by using ctx as audio"""
        previous_audio = audio[:, context_steps - 1 : -1]
        next_audio = audio[:, context_steps:]
        if self.audio_conditioning == "past_only":
            previous_audio = repeat(
                audio[:, context_steps - 1],
                "batch feature -> batch time feature",
                time=audio.shape[1] - context_steps,
            )
            next_audio = previous_audio
        return self(video[:, context_steps - 1 : -1], video[:, context_steps:], previous_audio, next_audio)

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        previous_video: Float[Tensor, "batch time patch video_feature"],
        next_video: Float[Tensor, "batch time patch video_feature"],
        previous_audio: Float[Tensor, "batch time audio_feature"],
        next_audio: Float[Tensor, "batch time audio_feature"],
    ) -> tuple[
        Float[Tensor, "batch time action"],
        Int[Tensor, "batch time"],
        Float[Tensor, "batch time action"],
        Float[Tensor, ""],
    ]:
        if previous_video.shape != next_video.shape or previous_audio.shape != next_audio.shape:
            raise ValueError("Latent-action transition pairs must have matching shapes")
        if previous_video.shape[:2] != previous_audio.shape[:2]:
            raise ValueError("Latent-action audio and video times must align")
        pooled_previous_video = reduce(previous_video, "batch time patch feature -> batch time feature", "mean")
        pooled_next_video = reduce(next_video, "batch time patch feature -> batch time feature", "mean")
        transition = torch.cat(
            (pooled_previous_video, pooled_next_video, previous_audio, next_audio),
            dim=-1,
        )
        encoded = self.encoder(transition)
        distances = reduce(
            (encoded[:, :, None] - self.codebook[None, None]).square(),
            "batch time code action -> batch time code",
            "sum",
        )
        code_ids = distances.argmin(dim=-1)
        selected_codes = self.codebook[code_ids]
        quantized = encoded + (selected_codes - encoded).detach()
        # TODO: ablate inclusion of commitment_loss, in general not fan of incl loss inline
        commitment_loss = F.mse_loss(encoded, selected_codes.detach())
        return quantized, code_ids, encoded, commitment_loss

    @torch.no_grad()
    @jaxtyped(typechecker=beartype)
    def update_codebook(
        self,
        encoded: Float[Tensor, "batch time action"],
        code_ids: Int[Tensor, "batch time"],
    ) -> None:
        assignments = F.one_hot(code_ids, self.code_count).to(encoded)
        counts = reduce(assignments, "batch time code -> code", "sum")
        sums = einsum(
            assignments,
            encoded,
            "batch time code, batch time action -> code action",
        )
        self.ema_counts.mul_(self.ema_decay).add_(counts, alpha=1.0 - self.ema_decay)
        self.ema_sums.mul_(self.ema_decay).add_(sums, alpha=1.0 - self.ema_decay)
        total_count = self.ema_counts.sum()
        normalized_counts = (
            (self.ema_counts + self.ema_epsilon) / (total_count + self.code_count * self.ema_epsilon) * total_count
        )
        self.codebook.copy_(self.ema_sums / normalized_counts[:, None])

    @jaxtyped(typechecker=beartype)
    def lookup(self, code_ids: Int[Tensor, "batch time"]) -> Float[Tensor, "batch time action"]:
        if torch.any(torch.logical_or(code_ids < 0, code_ids >= self.code_count)):
            raise ValueError(f"Latent action IDs must be in [0, {self.code_count})")
        return self.codebook[code_ids]
