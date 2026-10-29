import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchcodec.decoders import AudioDecoder, VideoDecoder
from torchcodec.transforms import Resize


@dataclass(frozen=True)
class PoolRecord:
    path: str
    dataset: str


def load_pool_records(catalog: Path, split: str) -> list[PoolRecord]:
    records = []
    for name in ("audioset_sources.jsonl", "existing_sources.jsonl"):
        with (catalog / name).open() as handle:
            for line in handle:
                entry = json.loads(line)
                if entry["source"]["split"] == split:
                    records.append(PoolRecord(entry["source"]["path"], entry["provenance"]["dataset"]))
    return records


class PoolWindowDataset(Dataset):
    def __init__(
        self,
        records: list[PoolRecord],
        windows: int,
        seed: int,
        total_frames: int,
        video_frames_per_step: int,
        action_dimension: int,
        image_size: int,
        video_rate: int,
        sample_rate: int,
        attempts: int = 16,
    ) -> None:
        self.paths = np.array([record.path for record in records])
        self.windows = windows
        self.seed = seed
        self.total_frames = total_frames
        self.time_steps = total_frames // video_frames_per_step
        self.action_dimension = action_dimension
        self.image_size = image_size
        self.video_rate = video_rate
        self.sample_rate = sample_rate
        self.attempts = attempts
        self.duration = total_frames / video_rate
        self.samples = round(self.duration * sample_rate)

    def __len__(self) -> int:
        return self.windows

    def read_window(self, path: str, generator: torch.Generator) -> tuple[Tensor, Tensor] | None:
        try:
            video = VideoDecoder(path, seek_mode="approximate", transforms=[Resize((self.image_size, self.image_size))])
            audio = AudioDecoder(path, sample_rate=self.sample_rate, num_channels=2)
            available = min(
                float(video.metadata.duration_seconds or 0.0), float(audio.metadata.duration_seconds or 0.0)
            )
            if available < self.duration + 0.2:
                return None
            start = 0.05 + float(torch.rand(1, generator=generator)) * (available - self.duration - 0.15)
            timestamps = [start + (frame + 1) / self.video_rate for frame in range(self.total_frames)]
            frames = video.get_frames_played_at(seconds=timestamps).data.float() / 255
            waveform = audio.get_samples_played_in_range(start_seconds=start, stop_seconds=start + self.duration).data
        except (RuntimeError, ValueError):
            return None
        if waveform.shape[-1] < self.samples:
            return None
        return frames, waveform[:, : self.samples]

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        generator = torch.Generator().manual_seed(self.seed * 1_000_003 + index)
        for attempt in range(self.attempts):
            path = str(self.paths[int(torch.randint(len(self.paths), (1,), generator=generator))])
            window = self.read_window(path, generator)
            if window is not None:
                frames, waveform = window
                return {
                    "frames": frames,
                    "waveform": waveform,
                    "actions": torch.zeros(self.time_steps, self.action_dimension),
                    "action_present": torch.tensor(0, dtype=torch.int64),
                    "example_seed": torch.tensor(index, dtype=torch.int64),
                }
        raise RuntimeError(f"No readable {self.duration:.2f}s audio-visual window after {self.attempts} records")
