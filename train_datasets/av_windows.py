from functools import lru_cache
from pathlib import Path
import torch
from torchcodec.decoders import AudioDecoder, VideoDecoder
from torchcodec.transforms import Resize
from train_datasets.corpus_types import CorpusWindow, DecodedAVWindow, MediaWindowConfig
from train_datasets.frame_indexes import read_frame_timeline
from train_datasets.input_staging import staged_media_path
from train_datasets.native_media_clock import (
    assert_native_video_clock,
    audio_grid_start,
)


@lru_cache(maxsize=2)
def decoders(
    video_path: str,
    audio_path: str,
    sample_rate: int,
    image_size: int,
    custom_frame_mappings: str | None = None,
    video_decoder_threads: int = 1,
) -> tuple[VideoDecoder, AudioDecoder]:
    return (
        VideoDecoder(
            video_path,
            num_ffmpeg_threads=video_decoder_threads,
            transforms=[Resize((image_size, image_size))],
            custom_frame_mappings=(None if custom_frame_mappings is None else Path(custom_frame_mappings).read_bytes()),
        ),
        AudioDecoder(audio_path, sample_rate=sample_rate, num_channels=2),
    )


def read_av_window(window: CorpusWindow, config: MediaWindowConfig) -> DecodedAVWindow:
    source = window.source
    assert source.split in {"train", "validation"}
    mapping = None
    if source.action_present and config.frame_index_directory is not None:
        relative = Path(source.source_id)
        assert not relative.is_absolute() and ".." not in relative.parts
        mapping = (config.frame_index_directory / relative).with_suffix(".json")
    timeline = None if mapping is None else read_frame_timeline(str(mapping))
    if timeline is not None:
        assert timeline.evidence.source_id == source.source_id and timeline.evidence.source_path == source.path
    media_path = staged_media_path(source, config.input_staging)
    video, audio = decoders(
        str(media_path),
        str(source.audio_path or media_path),
        config.sample_rate,
        config.image_size,
        None if mapping is None else str(mapping),
        video_decoder_threads=config.video_decoder_threads,
    )
    if timeline is not None:
        assert video.stream_index == timeline.evidence.stream_index
    assert audio.metadata.num_channels in {1, 2}
    start = audio_grid_start(float(audio.metadata.begin_stream_seconds), window.start_seconds, config.sample_rate)
    assert abs(start - window.start_seconds) <= 1 / config.sample_rate
    frame_count = config.context_frames + config.target_frames
    assert frame_count * config.sample_rate % config.video_rate == 0
    duration = frame_count / config.video_rate
    timestamps = [start + (frame + 1) / config.video_rate for frame in range(frame_count)]
    played = video.get_frames_played_at(seconds=timestamps)
    queries = torch.tensor(timestamps, dtype=played.pts_seconds.dtype)
    assert_native_video_clock(played.pts_seconds, played.duration_seconds, queries, config, timeline)
    samples = audio.get_samples_played_in_range(start_seconds=start, stop_seconds=start + duration)
    assert abs(samples.pts_seconds - start) <= 1 / config.sample_rate, (
        source.source_id,
        "audio_pts",
        samples.pts_seconds,
        "requested",
        start,
    )
    assert samples.data.shape == (2, frame_count * config.sample_rate // config.video_rate), (
        "native_audio_shape",
        source.source_id,
        tuple(samples.data.shape),
    )
    assert played.data.shape == (frame_count, 3, config.image_size, config.image_size)
    assert torch.isfinite(played.data).all() and torch.isfinite(samples.data).all()
    return DecodedAVWindow(played.data.float() / 255, samples.data, start)
