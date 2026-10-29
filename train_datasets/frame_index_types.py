from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from pydantic import NonNegativeInt, PositiveInt
from configs.config_types import FrozenModel


class NativeFramePacket(FrozenModel):
    pts: int
    duration: NonNegativeInt
    key_frame: Literal[0, 1]


class FrameIndexEvidence(FrozenModel):
    completed: Literal[True] = True
    source_id: str
    source_path: Path
    stream_index: NonNegativeInt
    time_base_numerator: PositiveInt
    time_base_denominator: PositiveInt
    frames: PositiveInt
    first_pts: int
    last_pts: int
    native_mapping_path: Path
    producer: Literal[
        "ffprobe native video packet PTS/duration/key-frame metadata",
        "PyAV native video packet PTS/duration/key-frame metadata",
    ] = "PyAV native video packet PTS/duration/key-frame metadata"
    durations_modified: Literal[False] = False


@dataclass(frozen=True)
class FrameTimeline:
    seconds: tuple[float, ...]
    end_seconds: float
    evidence: FrameIndexEvidence
