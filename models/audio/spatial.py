import torch
from beartype import beartype
from einops import rearrange, repeat
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn
from torchaudio.functional import melscale_fbanks


@jaxtyped(typechecker=beartype)
def ambisonic_microphones(
    ambisonics: Float[Tensor, "4 sample"],
) -> Float[Tensor, "4 sample"]:
    omni, left, up, front = ambisonics
    return torch.stack((omni + left, omni - left, up, front))


class BinauralFeatures(nn.Module):
    def __init__(
        self,
        sample_rate: int,
        frame_rate: int,
        mel_bands: int,
        maximum_interaural_delay_seconds: float,
        feature_dimension: int,
        ambisonic: bool = False,
    ) -> None:
        super().__init__()
        self.ambisonic = ambisonic
        self.mel_bands = mel_bands
        assert sample_rate % frame_rate == 0
        self.frame_samples = sample_rate // frame_rate
        self.lag_limit = round(maximum_interaural_delay_seconds * sample_rate)
        self.feature_dimension = feature_dimension
        assert 2 * self.lag_limit + 1 + mel_bands * (5 if ambisonic else 1) + 2 * int(ambisonic) <= feature_dimension
        self.register_buffer("window", torch.hann_window(self.frame_samples), persistent=False)
        self.register_buffer(
            "mel_filters",
            melscale_fbanks(
                self.frame_samples // 2 + 1,
                50.0,
                sample_rate / 2,
                mel_bands,
                sample_rate,
            ),
            persistent=False,
        )

    @jaxtyped(typechecker=beartype)
    def forward(self, waveform: Float[Tensor, "batch channel sample"]) -> Float[Tensor, "batch feature time"]:
        assert waveform.shape[1] == (4 if self.ambisonic else 2)
        with torch.autocast(waveform.device.type, enabled=False):
            return self.binaural(waveform.float())

    def binaural(self, waveform: Tensor) -> Tensor:
        frames = rearrange(
            waveform[..., : waveform.shape[-1] // self.frame_samples * self.frame_samples].float(),
            "batch channel (time sample) -> batch channel time sample",
            sample=self.frame_samples,
        )
        spectrum = torch.fft.rfft(frames * self.window, dim=-1)
        left, right = spectrum[:, 0], spectrum[:, 1]
        cross = right * left.conj()
        correlation = torch.fft.irfft(cross / cross.abs().clamp_min(1e-8), n=self.frame_samples, dim=-1)
        delays = torch.cat(
            (
                correlation[..., -self.lag_limit :],
                correlation[..., : self.lag_limit + 1],
            ),
            dim=-1,
        )
        band_energy = spectrum.abs().square() @ self.mel_filters
        level_difference = 10 * torch.log10((band_energy[:, 1] + 1e-8) / (band_energy[:, 0] + 1e-8))
        features = torch.cat((delays, level_difference.clamp(-40.0, 40.0)), dim=-1)
        if self.ambisonic:
            features = torch.cat((features, self.sound_field(spectrum)), dim=-1)
        padding = features.new_zeros(*features.shape[:-1], self.feature_dimension - features.shape[-1])
        return rearrange(
            torch.cat((features, padding), dim=-1),
            "batch time feature -> batch feature time",
        )

    def sound_field(self, spectrum: Tensor) -> Tensor:
        omni = (spectrum[:, 0] + spectrum[:, 1]) / 2
        axes = torch.stack(
            (spectrum[:, 3], (spectrum[:, 0] - spectrum[:, 1]) / 2, spectrum[:, 2]),
            dim=1,
        )
        intensity = (omni.conj()[:, None] * axes).real @ self.mel_filters
        energy = ((omni.abs().square() + axes.abs().square().sum(dim=1)) / 2) @ self.mel_filters
        magnitude = intensity.square().sum(dim=1).sqrt()
        direction = intensity / magnitude[:, None].clamp_min(1e-8)
        diffuseness = (1 - magnitude / energy.clamp_min(1e-8)).clamp(0.0, 1.0)
        present = (spectrum[:, 3].abs().square().sum(dim=(1, 2)) > 0).float()
        elevation = (spectrum[:, 2].abs().square().sum(dim=(1, 2)) > 0).float()
        field = torch.cat(
            (
                rearrange(
                    direction,
                    "batch component time band -> batch time (component band)",
                ),
                diffuseness,
            ),
            dim=-1,
        )
        flags = repeat(
            torch.stack((present, elevation), dim=-1),
            "batch flag -> batch time flag",
            time=field.shape[1],
        )
        return torch.cat((field * flags[..., :1], flags), dim=-1)
