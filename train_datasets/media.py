import re
from train_datasets.av_windows import read_av_window
from train_datasets.corpus_types import CorpusWindow, MediaWindowConfig, SourceRecord
from train_datasets.media_types import AudioVisualClip, MediaRecord


def load_av_clip(
    record: MediaRecord,
    frames_per_second: int,
    total_steps: int,
    image_size: int,
    sample_rate: int,
    input_channels: int,
) -> AudioVisualClip:
    assert input_channels == 2
    source = SourceRecord(
        source_id=record.sample_id,
        recording_id=record.recording_id or str(record.path.parent / re.sub(r"_split_?\d+$", "", record.path.stem)),
        dataset=record.dataset,
        path=record.path,
        controls_path=record.controls,
        action_present=record.controls is not None,
        split=record.split,
    )
    settings = MediaWindowConfig(
        image_size=image_size,
        video_rate=frames_per_second,
        sample_rate=sample_rate,
        context_frames=total_steps,
        target_frames=0,
    )
    decoded = read_av_window(CorpusWindow(source=source, start_seconds=record.start_seconds), settings)
    return AudioVisualClip(record, decoded.frames, decoded.waveform)
