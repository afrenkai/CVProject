import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchcodec.decoders import AudioDecoder, VideoDecoder
from torchcodec.transforms import Resize
from data.control_types import ControlLayout
from data.data_types import ActionWindowConfig, DatasetConfig
from data.recorded_capture import read_recorded_capture
from data.recorded_controls import controls_for_window
from models.audio.spatial import ambisonic_microphones
from models.encoder_types import FrozenEncoderConfig


@dataclass(frozen=True)
class PoolRecord:
    path: str
    dataset: str
    actions_path: str | None = None
    audio_path: str | None = None


def load_pool_records(catalog: Path, split: str) -> list[PoolRecord]:
    records = []
    for source in sorted(catalog.glob("*_sources.jsonl")):
        if source.name.startswith("excluded_"):
            continue
        with source.open() as handle:
            for line in handle:
                entry = json.loads(line)
                if entry["source"]["split"] == split:
                    records.append(
                        PoolRecord(
                            entry["source"]["path"],
                            entry["provenance"]["dataset"],
                            entry["source"].get("actions_path"),
                            entry["source"].get("audio_path"),
                        )
                    )
    return records


class PoolWindowDataset(Dataset):
    def __init__(
        self,
        records: list[PoolRecord],
        config: DatasetConfig,
        encoder: FrozenEncoderConfig,
        total_frames: int,
        layout: ControlLayout,
    ) -> None:
        self.config = config
        self.paths = np.array([record.path for record in records])
        self.audio_paths = np.array([record.audio_path or "" for record in records])
        self.action_paths = np.array([record.actions_path or "" for record in records])
        self.datasets = np.array([record.dataset for record in records])
        weights = config.group_weights or {
            name: float((self.datasets == name).sum()) for name in np.unique(self.datasets)
        }
        present = [name for name in weights if (self.datasets == name).any()]
        self.groups = [np.flatnonzero(self.datasets == name) for name in present]
        self.group_weights = torch.tensor([weights[name] for name in present])
        self.total_frames = total_frames
        self.time_steps = total_frames // encoder.video_frames_per_step
        self.action_dimension = layout.dimension
        self.image_size = encoder.image_size
        self.video_rate = encoder.video_rate
        self.sample_rate = encoder.loading_sample_rate
        self.waveform_channels = encoder.waveform_channels
        self.duration = total_frames / encoder.video_rate
        self.samples = round(self.duration * self.sample_rate)
        self.action_config = (
            ActionWindowConfig(
                context_frames=encoder.context_frames,
                target_frames=total_frames - encoder.context_frames,
                video_frames_per_step=encoder.video_frames_per_step,
                video_rate=encoder.video_rate,
                control_layout=layout,
            )
            if config.recorded_actions
            else None
        )

    def __len__(self) -> int:
        return self.config.windows

    def audio_decoder(self, record_index: int) -> AudioDecoder:
        audio_path = str(self.audio_paths[record_index])
        if audio_path and self.waveform_channels == 4:
            return AudioDecoder(audio_path, sample_rate=self.sample_rate)
        return AudioDecoder(str(self.paths[record_index]), sample_rate=self.sample_rate, num_channels=2)

    def waveform_layout(self, waveform: Tensor) -> Tensor:
        if waveform.shape[0] == 4:
            return ambisonic_microphones(waveform)
        assert waveform.shape[0] == 2
        return torch.cat((waveform, waveform.new_zeros(self.waveform_channels - 2, waveform.shape[-1])))

    def read_window(self, record_index: int, generator: torch.Generator) -> tuple[Tensor, Tensor, float] | None:
        try:
            video = VideoDecoder(
                str(self.paths[record_index]),
                seek_mode="approximate",
                transforms=[Resize((self.image_size, self.image_size))],
            )
            audio = self.audio_decoder(record_index)
            available = min(
                float(video.metadata.duration_seconds or 0.0), float(audio.metadata.duration_seconds or 0.0)
            )
            if available < self.duration + 0.2:
                return None
            start = 0.05 + float(torch.rand(1, generator=generator)) * (available - self.duration - 0.15)
            timestamps = [start + (frame + 1) / self.video_rate for frame in range(self.total_frames)]
            frames = video.get_frames_played_at(seconds=timestamps).data.float() / 255
            waveform = audio.get_samples_played_in_range(start_seconds=start, stop_seconds=start + self.duration).data
        except (RuntimeError, ValueError, IndexError):
            return None
        if waveform.shape[-1] < self.samples:
            return None
        return frames, self.waveform_layout(waveform[:, : self.samples]), start

    def recorded_actions(self, record_index: int, start: float) -> Tensor | None:
        assert self.action_config is not None
        try:
            capture = read_recorded_capture(
                Path(str(self.paths[record_index])),
                self.action_config,
                stop_seconds=start + self.duration + 0.5,
                start_seconds=max(0.0, start - self.action_config.control_state_lookback_seconds),
            )
            return controls_for_window(capture, start, self.action_config)
        except (AssertionError, OSError, ValueError):
            return None

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        source_index = index % self.config.distinct_windows if self.config.distinct_windows else index
        generator = torch.Generator().manual_seed(self.config.seed * 1_000_003 + source_index)
        for attempt in range(self.config.attempts):
            group = self.groups[int(torch.multinomial(self.group_weights, 1, generator=generator))]
            record_index = group[int(torch.randint(len(group), (1,), generator=generator))]
            window = self.read_window(record_index, generator)
            if window is None:
                continue
            frames, waveform, start = window
            actions = None
            if self.action_config is not None and self.action_paths[record_index]:
                actions = self.recorded_actions(record_index, start)
                if actions is None:
                    continue
            if actions is None and float(torch.rand(1, generator=generator)) < self.config.mirror_probability:
                frames = frames.flip(-1)
                waveform = waveform[[1, 0, *range(2, waveform.shape[0])]]
            return {
                "frames": frames,
                "waveform": waveform,
                "actions": torch.zeros(self.time_steps, self.action_dimension) if actions is None else actions,
                "action_present": torch.tensor(int(actions is not None), dtype=torch.int64),
            }
        raise RuntimeError(f"No readable {self.duration:.2f}s audio-visual window after {self.config.attempts} records")


class SyntheticWindowDataset(PoolWindowDataset):
    def read_window(self, record_index: int, generator: torch.Generator) -> tuple[Tensor, Tensor, float] | None:
        return (
            torch.rand(self.total_frames, 3, self.image_size, self.image_size, generator=generator),
            self.waveform_layout(0.1 * torch.randn(2, self.samples, generator=generator)),
            0.0,
        )


def build_dataset(
    config: DatasetConfig, encoder: FrozenEncoderConfig, total_frames: int, layout: ControlLayout
) -> PoolWindowDataset:
    if config.catalog == "synthetic":
        return SyntheticWindowDataset([PoolRecord("synthetic", "synthetic")], config, encoder, total_frames, layout)
    return PoolWindowDataset(load_pool_records(Path(config.catalog), "train"), config, encoder, total_frames, layout)


def build_validation_datasets(
    config: DatasetConfig, encoder: FrozenEncoderConfig, total_frames: int, layout: ControlLayout
) -> dict[str, PoolWindowDataset]:
    config = config.model_copy(
        update={
            "windows": config.validation_windows,
            "mirror_probability": 0.0,
            "distinct_windows": 0,
            "group_weights": None,
        }
    )
    if config.catalog == "synthetic":
        return {"synthetic": build_dataset(config, encoder, total_frames, layout)}
    records = load_pool_records(Path(config.catalog), "validation")
    names = sorted({record.dataset for record in records})
    return {
        name: PoolWindowDataset(
            [record for record in records if record.dataset == name], config, encoder, total_frames, layout
        )
        for name in names
    }
