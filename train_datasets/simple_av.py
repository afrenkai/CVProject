from dataclasses import dataclass
from pathlib import Path
import torch
import torch.nn.functional as F
from datasets import Dataset as HFDataset
from einops import repeat
from torch import Tensor
from torch.utils.data import Dataset
from torchcodec.decoders import AudioDecoder, VideoDecoder
from torchcodec.transforms import Resize


@dataclass(frozen=True)
class AVClip:
    path: Path
    start_seconds: float
    side: int = 0


def scan_video_windows(path: Path, total_frames: int, video_rate: int, stride_seconds: float) -> list[AVClip]:
    decoder = VideoDecoder(str(path))
    duration = float(decoder.metadata.duration_seconds)
    clip_duration = total_frames / video_rate
    clips = []
    start = 0.0
    while start + clip_duration <= duration:
        clips.append(AVClip(path=path, start_seconds=start))
        start += stride_seconds
    return clips


class SimpleAVDataset(Dataset):

    def __init__(
        self,
        clips: list[AVClip],
        total_frames: int,
        video_frames_per_step: int,
        action_dimension: int,
        image_size: int = 128,
        video_rate: int = 30,
        sample_rate: int = 16000,
    ) -> None:
        self.clips = clips
        self.total_frames = total_frames
        self.time_steps = total_frames // video_frames_per_step
        self.action_dimension = action_dimension
        self.image_size = image_size
        self.video_rate = video_rate
        self.sample_rate = sample_rate

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        clip = self.clips[index]
        duration = self.total_frames / self.video_rate
        timestamps = [clip.start_seconds + (f + 1) / self.video_rate for f in range(self.total_frames)]

        video = VideoDecoder(str(clip.path), transforms=[Resize((self.image_size, self.image_size))])
        audio = AudioDecoder(str(clip.path), sample_rate=self.sample_rate, num_channels=2)

        frames = video.get_frames_played_at(seconds=timestamps).data.float() / 255
        waveform = audio.get_samples_played_in_range(
            start_seconds=clip.start_seconds, stop_seconds=clip.start_seconds + duration
        ).data[:, : round(duration * self.sample_rate)]

        return {
            "frames": frames,
            "waveform": waveform,
            "actions": torch.zeros(self.time_steps, self.action_dimension),
            "action_present": torch.tensor(0, dtype=torch.int64),
            "example_seed": torch.tensor(index, dtype=torch.int64),
            "side": torch.tensor(clip.side, dtype=torch.int64),
        }


def scan_hf_windows(
    dataset: HFDataset, total_frames: int, video_rate: int, stride_seconds: float
) -> list[tuple[int, float]]:
    clip_duration = total_frames / video_rate
    windows = []
    for row in range(len(dataset)):
        duration = float(dataset[row]["video"].metadata.duration_seconds)
        start = 0.0
        while start + clip_duration < duration:
            windows.append((row, start))
            start += stride_seconds
    return windows


class HFAVDataset(Dataset):

    def __init__(
        self,
        dataset: HFDataset,
        windows: list[tuple[int, float]],
        total_frames: int,
        video_frames_per_step: int,
        action_dimension: int,
        image_size: int = 128,
        video_rate: int = 30,
    ) -> None:
        self.dataset = dataset
        self.windows = windows
        self.total_frames = total_frames
        self.time_steps = total_frames // video_frames_per_step
        self.action_dimension = action_dimension
        self.image_size = image_size
        self.video_rate = video_rate

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        row, start = self.windows[index]
        example = self.dataset[row]
        duration = self.total_frames / self.video_rate
        timestamps = [start + (f + 1) / self.video_rate for f in range(self.total_frames)]

        frames = example["video"].get_frames_played_at(seconds=timestamps).data.float() / 255
        frames = F.interpolate(frames, size=(self.image_size, self.image_size), mode="bilinear", antialias=True)
        waveform = example["audio"].get_samples_played_in_range(start_seconds=start, stop_seconds=start + duration).data
        if waveform.shape[0] == 1:
            waveform = repeat(waveform, "1 sample -> 2 sample")

        return {
            "frames": frames,
            "waveform": waveform,
            "actions": torch.zeros(self.time_steps, self.action_dimension),
            "action_present": torch.tensor(0, dtype=torch.int64),
            "example_seed": torch.tensor(index, dtype=torch.int64),
        }
