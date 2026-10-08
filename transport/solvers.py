import torch
from torchdiffeq import odeint
from torch import Tensor
import torch.nn as nn
from beartype import beartype
from einops import rearrange
from jaxtyping import Float, jaxtyped

from architecture.av_model import AudioVisualFlowPredictor

# initial code taken from https://github.com/facebookresearch/Flow-World-Models/blob/main/transport/integrators.py


def gradient(out: Tensor, x: Tensor, grad_out: Tensor = None, create_graph: bool=False, retain_graph: bool=False) -> torch.autograd.grad:
    if not grad_out:
        # we need to remove this output from the computational graph since it is not differentiable
        grad_out = torch.ones_like(out).detach()

    # we return the 1st derivative using autograd
    return torch.autograd.grad(
        out,
        x,
        grad_outputs = grad_out,
        create_graph=create_graph,
        retain_graph=retain_graph
    )[0]


class SDESolver:
    """
    solves an SDE of the form
    X_t = X_0 + \int_0^t \sigma_sdB_s + \int_0^t \mu_s ds
    assuming B is a brownian motion.

    Ito's lemma can be used to solve
    dX_t = \mu_t dt  + \sigma dB_t 
    where the drift term is the lhs (\mu) and the diffusion term is the rhs (\sigma)

    We can use the Euler Maruyama method https://en.wikipedia.org/wiki/Euler%E2%80%93Maruyama_method
    to approximate the numerical solution. Another method is the Heun Method, which can be used for ODEs or SDEs.
    """
    def __init__(
        self,
        drift_term,
        diffusion_term,
        *,
        t0,
        t1,
        n_steps,
        sampler: str,
        time_dist_offset
    ):
        assert t0 < t1, "SDE sampler not in reverse, i.e. we are doing Wiener process"

        self.timesteps = n_steps
        self.T = 1 - torch.linspace(t0, t1, n_steps)
        self.T = time_dist_offset * self.T / (1 + (time_dist_offset - 1 ) * self.T)
        self.drift = drift_term
        self.diffusion= diffusion_term
        self.sampler_type = sampler
        self.dist_time_shift = time_dist_offset

    def euler_maruyama_step(self, x: Tensor, mu_x: Tensor, t_now, t_next, model: nn.Module, **model_kwargs):
        B_t = torch.randn(x.size()).to(x)
        t = torch.ones(x.size(0)).to(x) * t_now
        dB = B_t * torch.sqrt(t_now - t_next)
        drift_term = self.drift(x, t, model, **model_kwargs)
        diffusion_term = self.diffusion(x, t)
        mu_x = x - drift_term * (t_now - t_next)
        x = mu_x + torch.sqrt(2 * diffusion_term) * dB
        return x, mu_x

        
    def heun_step(self, x, mu_x, t_now, t_next, model, **model_kwargs):
        B_t = torch.randn(x.size()).to(x)
        dw = B_t * torch.sqrt(t_now - t_next)
        diffusion = self.diffusion(x, torch.ones(x.size(0)).to(x) * t_now)
        xhat = x + torch.sqrt(2 * diffusion) * dw
        K1 = self.drift(xhat, torch.ones(x.size(0)).to(x) * t_now, model, **model_kwargs)
        xp = xhat - (t_now - t_next) * K1
        K2 = self.drift(xp, torch.ones(x.size(0)).to(x) * t_next, model, **model_kwargs)
        return xhat - 0.5 * (t_now - t_next) * (
            K1 + K2
        ), xhat  # at last time point we do not perform the heun step


    #TODO: refactor this. 0 reason why we need a dict if only 2 choices tbh
    def forward_fn(self):
        sampler_dict = {
            "euler": self.euler_maruyama_step,
            "heun": self.heun_step,
        }

        try:
            sampler = sampler_dict[self.sampler_type]
        except:
            raise NotImplementedError("Sampler type not implemented.")

        return sampler

    def sample(self, init, model, **model_kwargs) -> tuple[torch.Tensor]:
        x = init
        mu_x = init
        samples = []
        sampler = self.forward_fn()
        #TODO: rewrite, don't like this syntax and think it can be made cleaner but this is how the original code looked
        for t_curr, t_next in zip(self.T[:-1], self.T[1:]):
            with torch.no_grad():
                x, mu_x = sampler(x, mu_x, t_curr, t_next, model, **model_kwargs)
                samples.append(x)

        return samples


@jaxtyped(typechecker=beartype)
def audio_clock(
    time: Float[Tensor, "batch"],
    delay: float,
) -> tuple[Float[Tensor, "batch"], Float[Tensor, "batch"]]:
    """
    not sure I liek clock but tired and lazy

    @param time: float tensor which is batch shaped. created as a timestep embedder
    @param delay: how long to "delay" audio for , since we stream the av

    p = t-d/(1-d) in [0.,1.]

    d/dt = ((t>d) bool mask passed over time tensor) / 1.0 - d)

    Returns
        p, d/dt
    """
    progress = ((time - delay) / (1.0 - delay)).clamp(0.0, 1.0)
    derivative = (time > delay).to(time) / (1.0 - delay)
    return progress, derivative


@jaxtyped(typechecker=beartype)
def flow_map_jump(
    video_state: Float[Tensor, "batch time patch video_feature"],
    audio_state: Float[Tensor, "batch time audio_feature"],
    video_velocity: Float[Tensor, "batch time patch video_feature"],
    audio_velocity: Float[Tensor, "batch time audio_feature"],
    start_time: Float[Tensor, "batch"],
    end_time: Float[Tensor, "batch"],
) -> tuple[
    Float[Tensor, "batch time patch video_feature"],
    Float[Tensor, "batch time audio_feature"],
]:
    if torch.any(end_time < start_time):
        raise ValueError("Flow-map end time must not precede start time")
    time_gap = end_time - start_time
    return (
        video_state
        + rearrange(time_gap, "batch -> batch 1 1 1") * video_velocity,
        audio_state + rearrange(time_gap, "batch -> batch 1 1") * audio_velocity,
    )


@torch.no_grad()
@jaxtyped(typechecker=beartype)
def solve_flow_map(
    predictor: AudioVisualFlowPredictor,
    video_state: Float[Tensor, "batch time patch video_feature"],
    audio_state: Float[Tensor, "batch time audio_feature"],
    context_video: Float[Tensor, "batch context_time patch video_feature"],
    context_audio: Float[Tensor, "batch context_time audio_feature"],
    actions: Float[Tensor, "batch time action"] | None,
    integration_grid: Float[Tensor, "grid"],
    diagonal: bool,
) -> tuple[
    Float[Tensor, "batch time patch video_feature"],
    Float[Tensor, "batch time audio_feature"],
]:
    for integration_index in range(integration_grid.shape[0] - 1):
        start_time = integration_grid[integration_index].expand(video_state.shape[0])
        end_time = integration_grid[integration_index + 1].expand(video_state.shape[0])
        model_end_time = start_time if diagonal else end_time
        video_velocity, audio_velocity = predictor(
            video_state,
            audio_state,
            start_time,
            model_end_time,
            context_video,
            context_audio,
            actions,
        )
        video_state, audio_state = flow_map_jump(
            video_state,
            audio_state,
            video_velocity,
            audio_velocity,
            start_time,
            end_time,
        )
    return video_state, audio_state


@torch.no_grad()
@jaxtyped(typechecker=beartype)
def solve_sde(
    predictor: AudioVisualFlowPredictor,
    video_state: Float[Tensor, "batch time patch video_feature"],
    audio_state: Float[Tensor, "batch time audio_feature"],
    context_video: Float[Tensor, "batch context_time patch video_feature"],
    context_audio: Float[Tensor, "batch context_time audio_feature"],
    actions: Float[Tensor, "batch time action"] | None,
    integration_grid: Float[Tensor, "grid"],
    audio_delay: float,
    noise_standard_deviation: float,
    diffusion_scale: float,
) -> tuple[
    Float[Tensor, "batch time patch video_feature"],
    Float[Tensor, "batch time audio_feature"],
]:
    noise_variance = noise_standard_deviation**2
    epsilon = torch.finfo(video_state.dtype).eps
    for integration_index in range(integration_grid.shape[0] - 1):
        start_time = integration_grid[integration_index].expand(video_state.shape[0])
        end_time = integration_grid[integration_index + 1].expand(video_state.shape[0])
        video_velocity, audio_velocity = predictor(
            video_state,
            audio_state,
            start_time,
            start_time,
            context_video,
            context_audio,
            actions,
        )

        video_progress = rearrange(start_time, "batch -> batch 1 1 1")
        video_remaining = 1.0 - video_progress
        video_noise = video_state - video_progress * video_velocity
        video_score = -video_noise / (
            video_remaining.clamp_min(epsilon) * noise_variance
        )

        audio_progress, audio_derivative = audio_clock(start_time, audio_delay)
        audio_progress = rearrange(audio_progress, "batch -> batch 1 1")
        audio_derivative = rearrange(audio_derivative, "batch -> batch 1 1")
        audio_velocity_time = torch.where(
            audio_derivative > 0.0,
            audio_progress / audio_derivative.clamp_min(epsilon),
            torch.zeros_like(audio_derivative),
        )
        audio_remaining = 1.0 - audio_progress
        audio_noise = audio_state - audio_velocity_time * audio_velocity
        audio_score = -audio_noise / (
            audio_remaining.clamp_min(epsilon) * noise_variance
        )

        video_diffusion = diffusion_scale * video_remaining
        audio_diffusion = diffusion_scale * audio_remaining
        video_drift = video_velocity + 0.5 * video_diffusion.square() * video_score
        audio_drift = audio_velocity + 0.5 * audio_diffusion.square() * audio_score
        step_size = end_time - start_time
        video_step = rearrange(step_size, "batch -> batch 1 1 1")
        audio_step = rearrange(step_size, "batch -> batch 1 1")
        video_state = (
            video_state
            + video_step * video_drift
            + video_step.sqrt() * video_diffusion * torch.randn_like(video_state)
        )
        audio_state = (
            audio_state
            + audio_step * audio_drift
            + audio_step.sqrt() * audio_diffusion * torch.randn_like(audio_state)
        )
    return video_state, audio_state
