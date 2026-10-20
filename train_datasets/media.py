import json
import re
import subprocess
from pathlib import Path
import numpy as np
import torch
from einops import rearrange
from imageio_ffmpeg import get_ffmpeg_exe
from train_datasets.media_types import AudioVisualClip, MediaRecord


def held_out_domain(domain: str, excluded: list[str]) -> bool:
    normalized = "".join(character for character in domain.casefold() if character.isalnum())
    return any(
        normalized.startswith("".join(character for character in name.casefold() if character.isalnum()))
        for name in excluded
    )


def source_recording_identity(source: str) -> str:
    path = Path(source)
    return str(path.parent / re.sub(r"_split_?\d+$", "", path.stem))


def validate_pretraining_records(records: list[MediaRecord], excluded: list[str]) -> None:
    if any(held_out_domain(record.dataset, excluded) for record in records):
        raise ValueError("🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨DATA LEAKAGE FOUND 🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨")


def load_media_manifest(path: Path, split: str) -> list[MediaRecord]:
    records = [MediaRecord.model_validate(record) for record in json.loads(path.read_text())]
    selected = [record for record in records if record.split == split]
    if not selected:
        raise ValueError(f"No {split} clips in {path}")
    return selected


def validate_recording_split(manifest: Path) -> None:
    records = [MediaRecord.model_validate(record) for record in json.loads(manifest.read_text())]
    training_paths = {record.path for record in records if record.split == "train"}
    validation_paths = {record.path for record in records if record.split == "validation"}
    if training_paths & validation_paths:
        raise ValueError("A recording appears in both training and validation")
    training_recordings = {record.recording_id for record in records if record.split == "train" and record.recording_id}
    validation_recordings = {
        record.recording_id for record in records if record.split == "validation" and record.recording_id
    }
    if training_recordings & validation_recordings:
        raise ValueError("Segments of one source recording appear in both training and validation")


def load_av_clip(
    record: MediaRecord,
    frames_per_second: int,
    total_steps: int,
    image_size: int,
    sample_rate: int,
    input_channels: int,
) -> AudioVisualClip:
    """Each frame should be jointly sampled with an audio sample at 24khz assuming 30fps sampling"""
    if sample_rate % frames_per_second:
        raise ValueError("Audio samples per frame must be integral")
    common = [get_ffmpeg_exe(), "-v", "error", "-threads", "1", "-nostdin"]
    video = subprocess.run(
        [
            *common,
            "-ss",
            str(record.start_seconds + 1 / frames_per_second),
            "-i",
            str(record.path),
            "-map",
            "0:v:0",
            "-vf",
            f"fps={frames_per_second},scale={image_size}:{image_size}",
            "-frames:v",
            str(total_steps),
            "-pix_fmt",
            "rgb24",
            "-f",
            "rawvideo",
            "pipe:1",
        ],
        check=True,
        capture_output=True,
    ).stdout
    sample_count = total_steps * sample_rate // frames_per_second
    audio = subprocess.run(
        [
            *common,
            "-ss",
            str(record.start_seconds),
            "-i",
            str(record.path),
            "-map",
            "0:a:0",
            "-t",
            str(total_steps / frames_per_second),
            "-ar",
            str(sample_rate),
            "-ac",
            str(input_channels),
            "-f",
            "f32le",
            "pipe:1",
        ],
        check=True,
        capture_output=True,
    ).stdout
    if len(video) != total_steps * image_size * image_size * 3 or len(audio) != sample_count * input_channels * 4:
        raise ValueError(f"Incomplete AV clip: {record.sample_id}")
    frames = (
        rearrange(
            torch.from_numpy(np.frombuffer(video, dtype=np.uint8).copy()),
            "(time height width channel) -> time channel height width",
            time=total_steps,
            height=image_size,
            width=image_size,
            channel=3,
        ).float()
        / 255
    )
    waveform = rearrange(
        torch.from_numpy(np.frombuffer(audio, dtype=np.float32).copy()),
        "(sample channel) -> channel sample",
        channel=input_channels,
    )
    if not torch.isfinite(waveform).all() or not waveform.square().mean() > 0:
        raise ValueError(f"Missing or invalid audio: {record.sample_id}")
    return AudioVisualClip(record, frames, waveform)
