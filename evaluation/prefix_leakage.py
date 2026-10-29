import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor
from architecture.pretraining.encoder_types import FrozenEncoderConfig, PrefixEncoder
from evaluation.types import PrefixCheckConfig, PrefixIntervention


@torch.no_grad()
@jaxtyped(typechecker=beartype)
def assert_prefix_independence(
    encoder: PrefixEncoder,
    frames: Float[Tensor, "one frame video_channel height width"],
    waveform: Float[Tensor, "one audio_channel sample"],
    config: FrozenEncoderConfig,
    tolerances: PrefixCheckConfig,
    seed: int,
) -> list[PrefixIntervention]:
    context_steps = config.context_frames // config.video_frames_per_step
    context_samples = config.context_frames * config.sample_rate // config.video_rate
    assert frames.shape[1] > config.context_frames and waveform.shape[-1] > context_samples
    observed_video, observed_audio = encoder.encode_observed(frames, waveform)
    baseline_video, baseline_audio = encoder.encode(frames, waveform)
    torch.testing.assert_close(
        baseline_video[:, :context_steps],
        observed_video,
        rtol=tolerances.relative_tolerance,
        atol=tolerances.absolute_tolerance,
    )
    torch.testing.assert_close(
        baseline_audio[:, :context_steps],
        observed_audio,
        rtol=tolerances.relative_tolerance,
        atol=tolerances.absolute_tolerance,
    )
    generator = torch.Generator(device=frames.device).manual_seed(seed)
    future_frames = torch.rand(
        frames[:, config.context_frames :].shape,
        generator=generator,
        device=frames.device,
        dtype=frames.dtype,
    )
    future_waveform = 0.05 * torch.randn(
        waveform[..., context_samples:].shape,
        generator=generator,
        device=waveform.device,
        dtype=waveform.dtype,
    )
    results: list[PrefixIntervention] = []
    for intervention in ("future_video", "future_audio", "both_futures"):
        changed_frames, changed_waveform = frames.clone(), waveform.clone()
        if intervention in ("future_video", "both_futures"):
            changed_frames[:, config.context_frames :] = future_frames
            assert not torch.equal(changed_frames[:, config.context_frames :], frames[:, config.context_frames :])
        if intervention in ("future_audio", "both_futures"):
            changed_waveform[..., context_samples:] = future_waveform
            assert not torch.equal(changed_waveform[..., context_samples:], waveform[..., context_samples:])
        assert torch.equal(changed_frames[:, : config.context_frames], frames[:, : config.context_frames])
        assert torch.equal(changed_waveform[..., :context_samples], waveform[..., :context_samples])
        changed_video, changed_audio = encoder.encode(changed_frames, changed_waveform)
        torch.testing.assert_close(
            changed_video[:, :context_steps],
            observed_video,
            rtol=tolerances.relative_tolerance,
            atol=tolerances.absolute_tolerance,
            msg=f"Future leakage into observed video: {intervention}",
        )
        torch.testing.assert_close(
            changed_audio[:, :context_steps],
            observed_audio,
            rtol=tolerances.relative_tolerance,
            atol=tolerances.absolute_tolerance,
            msg=f"Future leakage into observed audio: {intervention}",
        )
        results.append(
            {
                "intervention": intervention,
                "video_maximum_absolute_difference": (changed_video[:, :context_steps] - observed_video)
                .abs()
                .max()
                .item(),
                "audio_maximum_absolute_difference": (changed_audio[:, :context_steps] - observed_audio)
                .abs()
                .max()
                .item(),
                "passed": True,
            }
        )
    return results
