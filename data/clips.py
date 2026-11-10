import io
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import fsspec
import numpy as np
import pyarrow.parquet
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchcodec.decoders import AudioDecoder, VideoDecoder
from torchcodec.transforms import Resize
from data.control_types import ControlLayout
from data.data_types import ActionClipConfig, DatasetConfig, SourceConfig
from data.opencs2_controls import opencs2_controls, read_opencs2_ticks
from data.recorded_capture import read_recorded_capture
from data.recorded_controls import controls_for_clip
from models.audio.spatial import ambisonic_microphones
from models.encoder_types import FrozenEncoderConfig


@dataclass(frozen=True)
class AVExample:
    media_path: str
    dataset: str
    actions_path: str | None = None
    audio_path: str | None = None
    ambisonic_path: str | None = None
    media_span: tuple[int, int] | None = None
    actions_span: tuple[int, int] | None = None


def read_span(path: str, offset: int, size: int) -> bytes:
    with fsspec.open(path, "rb") as handle:
        handle.seek(offset)
        return handle.read(size)


def shared_bounds(audio: AudioDecoder, video: VideoDecoder) -> tuple[float, float]:
    audio_begin = float(audio.metadata.begin_stream_seconds or 0.0)
    video_begin = float(video.metadata.begin_stream_seconds or 0.0)
    audio_end = audio_begin + float(audio.metadata.duration_seconds or 0.0)
    video_end = float(video.metadata.end_stream_seconds or video_begin + float(video.metadata.duration_seconds or 0.0))
    return max(audio_begin, video_begin), min(audio_end, video_end)


def has_split(source: SourceConfig, split: str) -> bool:
    filesystem, path = fsspec.core.url_to_fs(source.splits.format(split=split))
    return filesystem.exists(path)


def shard_examples(name: str, source: SourceConfig, samples: str, keys: list[str]) -> list[AVExample]:
    columns = ["media_id", "shard_path", "mp4_offset", "mp4_size", "ticks_offset", "ticks_size"]
    with fsspec.open(samples, "rb") as handle:
        table = pyarrow.parquet.read_table(handle, columns=columns).to_pandas().set_index("media_id")
    return [
        AVExample(
            f"{source.root}/{row.shard_path}",
            name,
            f"{source.root}/{row.shard_path}",
            media_span=(row.mp4_offset, row.mp4_size),
            actions_span=(row.ticks_offset, row.ticks_size),
        )
        for row in table.loc[keys].itertuples()
    ]


def source_examples(name: str, source: SourceConfig, split: str) -> list[AVExample]:
    with fsspec.open(source.splits.format(split=split), "rt") as handle:
        keys = handle.read().split()
    if source.samples is not None:
        return shard_examples(name, source, source.samples, keys)
    assert source.media is not None, f"{name} needs either a media template or a samples index"
    return [
        AVExample(
            source.media.format(key=key),
            name,
            None if source.actions is None else source.actions.format(key=key),
            None if source.audio is None else source.audio.format(key=key),
            None if source.ambisonic is None else source.ambisonic.format(key=key),
        )
        for key in keys
    ]


class AVDataset(Dataset, ABC):
    def __init__(self, examples: list[AVExample], weights: dict[str, float] | None, config: DatasetConfig) -> None:
        self.config = config
        self.media_paths = np.array([example.media_path for example in examples])
        self.audio_paths = np.array([example.audio_path or "" for example in examples])
        self.ambisonic_paths = np.array([example.ambisonic_path or "" for example in examples])
        self.action_paths = np.array([example.actions_path or "" for example in examples])
        self.media_spans = np.array([example.media_span or (-1, 0) for example in examples], dtype=np.int64)
        self.action_spans = np.array([example.actions_span or (-1, 0) for example in examples], dtype=np.int64)
        self.datasets = np.array([example.dataset for example in examples])
        weights = weights or {name: float((self.datasets == name).sum()) for name in np.unique(self.datasets)}
        present = [name for name in weights if (self.datasets == name).any()]
        self.groups = [np.flatnonzero(self.datasets == name) for name in present]
        self.group_weights = torch.tensor([weights[name] for name in present])

    def __len__(self) -> int:
        return self.config.clips

    def media(self, example_index: int) -> str | bytes:
        path = str(self.media_paths[example_index])
        offset, size = self.media_spans[example_index]
        return path if offset < 0 else read_span(path, int(offset), int(size))

    def audio_source(self, example_index: int, media: str | bytes) -> str | bytes:
        return str(self.audio_paths[example_index]) or media

    def video_decoder(self, media: str | bytes, size: int) -> VideoDecoder:
        return VideoDecoder(media, seek_mode="approximate", transforms=[Resize((size, size))])

    @abstractmethod
    def sample(self, example_index: int, generator: torch.Generator) -> dict[str, Tensor] | None: ...

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        generator = torch.Generator().manual_seed(self.config.seed * 1_000_067 + index)
        for attempt in range(self.config.attempts):
            group = self.groups[int(torch.multinomial(self.group_weights, 1, generator=generator))]
            example_index = group[int(torch.randint(len(group), (1,), generator=generator))]
            sample = self.sample(example_index, generator)
            if sample is not None:
                return sample
        raise RuntimeError(f"No readable clip after {self.config.attempts} examples")


class WorldModelDataset(AVDataset):
    def __init__(
        self,
        examples: list[AVExample],
        weights: dict[str, float] | None,
        config: DatasetConfig,
        encoder: FrozenEncoderConfig,
        total_frames: int,
        layout: ControlLayout,
    ) -> None:
        super().__init__(examples, weights, config)
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
            ActionClipConfig(
                context_frames=encoder.context_frames,
                target_frames=total_frames - encoder.context_frames,
                video_frames_per_step=encoder.video_frames_per_step,
                video_rate=encoder.video_rate,
                control_layout=layout,
            )
            if config.recorded_actions
            else None
        )

    def audio_decoder(self, example_index: int, media: str | bytes) -> AudioDecoder:
        ambisonic_path = str(self.ambisonic_paths[example_index])
        if ambisonic_path and self.waveform_channels == 4:
            return AudioDecoder(ambisonic_path, sample_rate=self.sample_rate)
        return AudioDecoder(self.audio_source(example_index, media), sample_rate=self.sample_rate, num_channels=2)

    def waveform_layout(self, waveform: Tensor) -> Tensor:
        if waveform.shape[0] == 4:
            return ambisonic_microphones(waveform)
        assert waveform.shape[0] == 2
        return torch.cat((waveform, waveform.new_zeros(self.waveform_channels - 2, waveform.shape[-1])))

    def read_clip(self, example_index: int, generator: torch.Generator) -> tuple[Tensor, Tensor, float] | None:
        try:
            media = self.media(example_index)
            video = self.video_decoder(media, self.image_size)
            audio = self.audio_decoder(example_index, media)
            begin, end = shared_bounds(audio, video)
            available = end - begin
            if available < self.duration + 0.2:
                return None
            start = begin + 0.05 + float(torch.rand(1, generator=generator)) * (available - self.duration - 0.15)
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

    def recorded_actions(self, example_index: int, start: float) -> Tensor | None:
        assert self.action_config is not None
        actions_path = str(self.action_paths[example_index])
        offset, size = self.action_spans[example_index]
        try:
            if offset >= 0:
                ticks = read_opencs2_ticks(io.BytesIO(read_span(actions_path, int(offset), int(size))))
                return opencs2_controls(ticks, start, self.action_config, self.action_config.opencs2_inputs)
            capture = read_recorded_capture(
                Path(str(self.media_paths[example_index])),
                self.action_config,
                stop_seconds=start + self.duration + 0.5,
                start_seconds=max(0.0, start - self.action_config.control_state_lookback_seconds),
            )
            return controls_for_clip(capture, start, self.action_config)
        except (AssertionError, OSError, ValueError):
            return None

    def sample(self, example_index: int, generator: torch.Generator) -> dict[str, Tensor] | None:
        clip = self.read_clip(example_index, generator)
        if clip is None:
            return None
        frames, waveform, start = clip
        actions = None
        if self.action_config is not None and self.action_paths[example_index]:
            actions = self.recorded_actions(example_index, start)
            if actions is None:
                return None
        if actions is None and float(torch.rand(1, generator=generator)) < self.config.mirror_probability:
            # thank god flip works along -dims
            frames = frames.flip(-1)
            waveform = waveform[[1, 0, *range(2, waveform.shape[0])]]
        return {
            "frames": frames,
            "waveform": waveform,
            "actions": torch.zeros(self.time_steps, self.action_dimension) if actions is None else actions,
            "action_present": torch.tensor(int(actions is not None), dtype=torch.int64),
        }


# this is more for CPU sanity checks, but we can also use it for "smoke tests" on GPU I guess
class SyntheticWorldModelDataset(WorldModelDataset):
    def read_clip(self, example_index: int, generator: torch.Generator) -> tuple[Tensor, Tensor, float] | None:
        return (
            torch.rand(self.total_frames, 3, self.image_size, self.image_size, generator=generator),
            self.waveform_layout(0.1 * torch.randn(2, self.history_samples + self.samples, generator=generator)),
            0.0,
        )


def training_examples(config: DatasetConfig) -> tuple[list[AVExample], dict[str, float]]:
    weights = {name: source.weight for name, source in config.sources.items() if source.weight > 0}
    return [example for name in weights for example in source_examples(name, config.sources[name], "train")], weights


def build_dataset(
    config: DatasetConfig, encoder: FrozenEncoderConfig, total_frames: int, layout: ControlLayout
) -> WorldModelDataset:
    if config.synthetic:
        return SyntheticWorldModelDataset(
            [AVExample("synthetic", "synthetic")], None, config, encoder, total_frames, layout
        )
    return WorldModelDataset(*training_examples(config), config, encoder, total_frames, layout)


def build_evaluation_datasets(
    config: DatasetConfig, split: str, encoder: FrozenEncoderConfig, total_frames: int, layout: ControlLayout
) -> dict[str, WorldModelDataset]:
    config = config.model_copy(update={"clips": config.validation_clips, "mirror_probability": 0.0})
    if config.synthetic:
        return {"synthetic": build_dataset(config, encoder, total_frames, layout)}
    return {
        name: WorldModelDataset(source_examples(name, source, split), None, config, encoder, total_frames, layout)
        for name, source in config.sources.items()
        if has_split(source, split)
    }
