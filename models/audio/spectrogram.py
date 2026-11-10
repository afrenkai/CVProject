import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor
from torchaudio.compliance.kaldi import fbank


def kaldi_fbank(
    waveform: Float[Tensor, "clip sample"], sample_rate: int, mel_bins: int, frame_shift: float
) -> Float[Tensor, "clip frame mel"]:
    with torch.autocast(waveform.device.type, enabled=False):
        return torch.stack(
            [
                fbank(
                    rearrange(clip, "sample -> 1 sample").float(),
                    htk_compat=True,
                    sample_frequency=sample_rate,
                    use_energy=False,
                    window_type="hanning",
                    num_mel_bins=mel_bins,
                    dither=0.0,
                    frame_shift=frame_shift,
                )
                for clip in waveform
            ]
        )
