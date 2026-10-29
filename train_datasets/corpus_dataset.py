import torch
from torch import Tensor
from torch.utils.data import Dataset
from train_datasets.av_windows import read_av_window
from train_datasets.corpus_types import CorpusConfig, CorpusWindow
from train_datasets.recorded_capture import read_recorded_capture
from train_datasets.recorded_controls import controls_for_window


def read_corpus_window(window: CorpusWindow, config: CorpusConfig) -> dict[str, Tensor]:
    source = window.source
    decoded = read_av_window(window, config)
    frame_count = config.context_frames + config.target_frames
    assert frame_count % config.video_frames_per_step == 0
    if source.action_present:
        assert source.actions_path == source.path.with_suffix(".mcap")
        capture = read_recorded_capture(
            source.path,
            config,
            stop_seconds=decoded.start_seconds + frame_count / config.video_rate + 0.5,
            start_seconds=max(0.0, decoded.start_seconds - config.control_state_lookback_seconds),
        )
        actions = controls_for_window(capture, decoded.start_seconds, config)
    else:
        assert source.actions_path is None
        actions = torch.zeros(frame_count // config.video_frames_per_step, config.control_layout.dimension)
    batch = {
        "frames": decoded.frames,
        "waveform": decoded.waveform,
        "actions": actions,
        "action_present": torch.tensor(int(source.action_present), dtype=torch.int64),
        "example_seed": torch.tensor(window.example_seed, dtype=torch.int64),
    }
    assert actions.shape == (frame_count // config.video_frames_per_step, config.control_layout.dimension)
    assert all(torch.isfinite(value).all() for value in batch.values())
    return batch


class CorpusWindows(Dataset[dict[str, Tensor]]):
    def __init__(self, windows: list[CorpusWindow], config: CorpusConfig) -> None:
        assert windows and all(window.source.split in {"train", "validation"} for window in windows)
        self.windows = windows
        self.config = config

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        return read_corpus_window(self.windows[index], self.config)
