import functools
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO
import fsspec
import numpy as np
import pyarrow.parquet
import torch
from fsspec import AbstractFileSystem
from torch import Tensor
from torch.utils.data import Dataset
from torchcodec.decoders import AudioDecoder, VideoDecoder
from torchcodec.transforms import Resize
from data.control_types import ControlLayout
from data.data_types import ActionWindowConfig, DatasetConfig, SourceConfig
from data.opencs2_controls import opencs2_controls, read_opencs2_ticks
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


@functools.lru_cache(maxsize=64)
def shard_filesystem(shard: str) -> AbstractFileSystem:
    protocol, path = fsspec.core.split_protocol(shard)
    return fsspec.filesystem("tar", fo=path, target_protocol=protocol or "file")


def open_member(url: str) -> BinaryIO:
    member, shard = url.removeprefix("tar://").split("::", 1)
    return shard_filesystem(shard).open(member, "rb")


def has_split(source: SourceConfig, split: str) -> bool:
    filesystem, path = fsspec.core.url_to_fs(source.splits.format(split=split))
    return filesystem.exists(path)


def shard_records(name: str, source: SourceConfig, samples: str, keys: list[str]) -> list[PoolRecord]:
    columns = ["media_id", "shard_path", "mp4_member", "ticks_member"]
    with fsspec.open(samples, "rb") as handle:
        table = pyarrow.parquet.read_table(handle, columns=columns).to_pandas().set_index("media_id")
    return [
        PoolRecord(
            f"tar://{row.mp4_member}::{source.root}/{row.shard_path}",
            name,
            f"tar://{row.ticks_member}::{source.root}/{row.shard_path}",
        )
        for row in table.loc[keys].itertuples()
    ]


def source_records(name: str, source: SourceConfig, split: str) -> list[PoolRecord]:
    with fsspec.open(source.splits.format(split=split), "rt") as handle:
        keys = handle.read().split()
    if source.samples is not None:
        return shard_records(name, source, source.samples, keys)
    assert source.video is not None, f"{name} needs either a video template or a samples index"
    return [
        PoolRecord(
            source.video.format(key=key),
            name,
            None if source.actions is None else source.actions.format(key=key),
            None if source.audio is None else source.audio.format(key=key),
        )
        for key in keys
    ]


class PoolWindowDataset(Dataset):
    def __init__(
        self,
        records: list[PoolRecord],
        weights: dict[str, float] | None,
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
        weights = weights or {name: float((self.datasets == name).sum()) for name in np.unique(self.datasets)}
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
        self.history_samples = encoder.audio_history_samples
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

    def media(self, record_index: int) -> str | BinaryIO:
        path = str(self.paths[record_index])
        return open_member(path) if "::" in path else path

    def video_decoder(self, record_index: int) -> VideoDecoder:
        return VideoDecoder(
            self.media(record_index),
            seek_mode="approximate",
            transforms=[Resize((self.image_size, self.image_size))],
        )

    def audio_decoder(self, record_index: int) -> AudioDecoder:
        audio_path = str(self.audio_paths[record_index])
        if audio_path and self.waveform_channels == 4:
            return AudioDecoder(audio_path, sample_rate=self.sample_rate)
        try:
            return AudioDecoder(self.media(record_index), sample_rate=self.sample_rate, num_channels=2)
        except ValueError:
            if not audio_path:
                raise
            return AudioDecoder(audio_path, sample_rate=self.sample_rate, num_channels=2)

    def waveform_layout(self, waveform: Tensor) -> Tensor:
        if waveform.shape[0] == 4:
            return ambisonic_microphones(waveform)
        assert waveform.shape[0] == 2
        return torch.cat((waveform, waveform.new_zeros(self.waveform_channels - 2, waveform.shape[-1])))

    def read_window(self, record_index: int, generator: torch.Generator) -> tuple[Tensor, Tensor, float] | None:
        try:
            video = self.video_decoder(record_index)
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
        history = self.audio_history(audio, start, waveform.shape[0])
        return frames, self.waveform_layout(torch.cat((history, waveform[:, : self.samples]), dim=-1)), start

    def audio_history(self, audio: AudioDecoder, start: float, channels: int) -> Tensor:
        if not self.history_samples or start <= 0:
            return torch.zeros(channels, self.history_samples)
        try:
            history = audio.get_samples_played_in_range(
                start_seconds=max(0.0, start - self.history_samples / self.sample_rate), stop_seconds=start
            ).data[:, -self.history_samples :]
        except (RuntimeError, ValueError, IndexError):
            history = torch.zeros(channels, 0)
        return torch.nn.functional.pad(history, (self.history_samples - history.shape[-1], 0))

    def recorded_actions(self, record_index: int, start: float) -> Tensor | None:
        assert self.action_config is not None
        actions_path = str(self.action_paths[record_index])
        try:
            if actions_path.split("::")[0].endswith(".parquet"):
                ticks = read_opencs2_ticks(open_member(actions_path))
                return opencs2_controls(ticks, start, self.action_config, self.action_config.opencs2_inputs)
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
        generator = torch.Generator().manual_seed(self.config.seed * 1_000_003 + index)
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


# this is more for CPU sanity checks, but we can also use it for "smoke tests" on GPU I guess
class SyntheticWindowDataset(PoolWindowDataset):
    def read_window(self, record_index: int, generator: torch.Generator) -> tuple[Tensor, Tensor, float] | None:
        return (
            torch.rand(self.total_frames, 3, self.image_size, self.image_size, generator=generator),
            self.waveform_layout(0.1 * torch.randn(2, self.history_samples + self.samples, generator=generator)),
            0.0,
        )


def build_dataset(
    config: DatasetConfig, encoder: FrozenEncoderConfig, total_frames: int, layout: ControlLayout
) -> PoolWindowDataset:
    if config.synthetic:
        return SyntheticWindowDataset(
            [PoolRecord("synthetic", "synthetic")], None, config, encoder, total_frames, layout
        )
    weights = {name: source.weight for name, source in config.sources.items() if source.weight > 0}
    records = [record for name in weights for record in source_records(name, config.sources[name], "train")]
    return PoolWindowDataset(records, weights, config, encoder, total_frames, layout)


def build_evaluation_datasets(
    config: DatasetConfig, split: str, encoder: FrozenEncoderConfig, total_frames: int, layout: ControlLayout
) -> dict[str, PoolWindowDataset]:
    config = config.model_copy(update={"windows": config.validation_windows, "mirror_probability": 0.0})
    if config.synthetic:
        return {"synthetic": build_dataset(config, encoder, total_frames, layout)}
    return {
        name: PoolWindowDataset(source_records(name, source, split), None, config, encoder, total_frames, layout)
        for name, source in config.sources.items()
        if has_split(source, split)
    }
