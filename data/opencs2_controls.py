from typing import BinaryIO
import numpy as np
import pyarrow.parquet
import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor
from data.control_types import Opencs2InputMap
from data.data_types import ActionWindowConfig


def read_opencs2_ticks(handle: BinaryIO) -> pyarrow.Table:
    with handle:
        return pyarrow.parquet.read_table(handle, columns=["t", "active", "delta_pitch", "delta_yaw", "is_alive"])


@jaxtyped(typechecker=beartype)
def opencs2_controls(
    ticks: pyarrow.Table, start_seconds: float, settings: ActionWindowConfig, inputs: Opencs2InputMap
) -> Float[Tensor, "block control"]:
    layout = settings.control_layout
    block_seconds = settings.video_frames_per_step / settings.video_rate
    block_count = (settings.context_frames + settings.target_frames) // settings.video_frames_per_step
    blocks = np.floor((ticks["t"].to_numpy() - start_seconds) / block_seconds).astype(np.int64)
    inside = np.flatnonzero((blocks >= 0) & (blocks < block_count))
    counts = np.bincount(blocks[inside], minlength=block_count)
    assert counts.min() >= inputs.minimum_ticks_per_block, "Tick stream does not cover the window"
    assert ticks["is_alive"].to_numpy()[inside].all(), "Player is not alive for the whole window"
    held_count = layout.keyboard_count + len(layout.mouse_buttons)
    controls = torch.zeros(block_count, layout.dimension, dtype=torch.float64)
    active = ticks["active"].to_pylist()
    for tick in inside:
        for name in active[tick]:
            for key in inputs.keys.get(name, ()):
                controls[blocks[tick], key] += 1
            if name in inputs.buttons:
                controls[blocks[tick], layout.keyboard_count + layout.mouse_buttons.index(inputs.buttons[name])] += 1
    controls[:, :held_count] /= torch.from_numpy(counts).unsqueeze(-1)
    yaw = (ticks["delta_yaw"].to_numpy()[inside] + 180.0) % 360.0 - 180.0
    pitch = ticks["delta_pitch"].to_numpy()[inside]
    mouse = np.zeros((block_count, 2))
    np.add.at(mouse, blocks[inside], np.stack((-yaw, pitch), axis=-1) / inputs.degrees_per_count)
    controls[:, held_count : held_count + 2] = torch.from_numpy(mouse)
    return controls.float()
