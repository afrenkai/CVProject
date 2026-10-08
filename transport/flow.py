from typing import Self
import torch
from beartype import beartype
from einops import rearrange
from jaxtyping import Float, jaxtyped
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from transport.solvers import audio_clock
from transport.types import AudioVisualInterpolant, TransportConfig


class AudioVisualFlowTransport:
    def __init__(self, config: TransportConfig) -> None:
        self.config = config

    @classmethod
    def from_omegaconf(cls, config: DictConfig) -> Self:
        values = OmegaConf.to_container(config, resolve=True)
        return cls(TransportConfig.model_validate(values))

    @jaxtyped(typechecker=beartype)
    def shift_time(self, time: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        shift = self.config.time_distribution.shift
        return shift * time / (1.0 + (shift - 1.0) * time)

    @jaxtyped(typechecker=beartype)
    def sample_time(self, reference: Float[Tensor, "batch ..."]) -> Float[Tensor, "batch"]:
        """
        given reference tensor and distribution (passed by cfg),
        construct a time-series following that reference. 
        similar to torch/np.linspace but dist instead of axes
        returns a sigmoid offset by the 1st and 2nd moments of the dist.
        helpful docs are torch.randn/rand for normal and unif, respectively. 
        """
        distribution = self.config.time_distribution
        if distribution.type == "uniform":
            time = torch.rand(
                reference.shape[0],
                device=reference.device,
                dtype=reference.dtype,
            )
        else:
            normal = torch.randn(
                reference.shape[0],
                device=reference.device,
                dtype=reference.dtype,
            )
            time = torch.sigmoid(distribution.mean + distribution.standard_deviation * normal)
        return self.shift_time(time)

    @jaxtyped(typechecker=beartype)
    def sample_interpolant(
        self,
        target_video: Float[Tensor, "batch time patch video_feature"],
        target_audio: Float[Tensor, "batch time audio_feature"],
    ) -> AudioVisualInterpolant:
        if target_video.shape[:2] != target_audio.shape[:2]:
            raise ValueError("Transport audio and video targets must align")
        time = self.sample_time(target_video)
        # we need tensor shapes to match, same as np.reshape(reference_tensor[-1])
        video_time = rearrange(time, "batch -> batch 1 1 1")
        audio_time, audio_derivative = audio_clock(time, self.config.audio_delay)
        # see above comment
        audio_time = rearrange(audio_time, "batch -> batch 1 1")
        audio_derivative = rearrange(audio_derivative, "batch -> batch 1 1")
        # subject to change, but usually a constant sigma is ok, we aren't doing VP/VESDE here
        # TODO: investigate if VP/VE SDE with a min and max sigma performs better here
        video_noise = self.config.noise_standard_deviation * torch.randn_like(target_video)
        audio_noise = self.config.noise_standard_deviation * torch.randn_like(target_audio)
        # SI formula: x + (t-s)v_{s,t}(x)
        return AudioVisualInterpolant(
            video_state=(1.0 - video_time) * video_noise + video_time * target_video,
            audio_state=(1.0 - audio_time) * audio_noise + audio_time * target_audio,
            video_velocity=target_video - video_noise,
            audio_velocity=audio_derivative * (target_audio - audio_noise),
            time=time,
        )

    @jaxtyped(typechecker=beartype)
    def sample_end_time(
        self,
        start_time: Float[Tensor, "batch"],
        maximum_time_gap: float,
    ) -> Float[Tensor, "batch"]:
        if not 0.0 < maximum_time_gap <= 1.0:
            raise ValueError("Maximum flow-map time gap must be in (0, 1]")
        return start_time + torch.rand_like(start_time) * (1.0 - start_time) * maximum_time_gap

    # what we call diag_frac. see fig 2  in https://arxiv.org/abs/2505.18825v2 for a nice visual
    @jaxtyped(typechecker=beartype)
    def time_grid(
        self,
        integration_steps: int,
        reference: Float[Tensor, "batch ..."],
    ) -> Float[Tensor, "grid"]:
        if integration_steps <= 0:
            raise ValueError("Flow-map integration steps must be positive")
        uniform_grid = torch.linspace(
            0.0,
            1.0,
            integration_steps + 1,
            device=reference.device,
            dtype=reference.dtype,
        )
        return self.shift_time(uniform_grid)
