import json
from functools import lru_cache
from pathlib import Path
import yaml
from train_datasets.frame_index_types import (
    FrameIndexEvidence,
    FrameTimeline,
    NativeFramePacket,
)


@lru_cache(maxsize=4)
def read_frame_timeline(path: str) -> FrameTimeline:
    mapping = Path(path)
    evidence = FrameIndexEvidence.model_validate(yaml.safe_load(mapping.with_suffix(".yaml").read_text()))
    packets = [NativeFramePacket.model_validate(row) for row in json.loads(mapping.read_text())["frames"]]
    assert (
        len(packets) == evidence.frames
        and packets[0].pts == evidence.first_pts
        and packets[-1].pts == evidence.last_pts
    )
    assert all(packets[index].pts < packets[index + 1].pts for index in range(len(packets) - 1))
    numerator = evidence.time_base_numerator
    denominator = evidence.time_base_denominator
    seconds = tuple(packet.pts * numerator / denominator for packet in packets)
    end = (packets[-1].pts + packets[-1].duration) * numerator / denominator
    return FrameTimeline(seconds, end, evidence)
