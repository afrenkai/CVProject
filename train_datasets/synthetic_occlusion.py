import math
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor
from torch.utils.data import Dataset


def render_background(image_size: int, generator: torch.Generator) -> Tensor:
    coarse = torch.rand(3, 6, 6, generator=generator)
    texture = F.interpolate(coarse[None], size=(image_size, image_size), mode="bicubic", align_corners=False)[0]
    base = torch.tensor([0.42, 0.45, 0.40])[:, None, None]
    return (base + 0.12 * (texture - 0.5)).clamp(0.0, 1.0)


def footstep(sample_rate: int, timbre: float, generator: torch.Generator) -> Tensor:
    duration = int(0.12 * sample_rate)
    seconds = torch.arange(duration) / sample_rate
    burst = torch.randn(duration, generator=generator) * torch.exp(-seconds / (0.012 + 0.02 * timbre))
    thump = torch.sin(2 * math.pi * (70.0 + 60.0 * timbre) * seconds) * torch.exp(-seconds / 0.04)
    return 0.6 * burst + 0.8 * thump


class SyntheticOcclusionDataset(Dataset):
    def __init__(
        self,
        examples: int,
        seed: int,
        total_frames: int = 42,
        video_frames_per_step: int = 3,
        action_dimension: int = 1,
        image_size: int = 112,
        video_rate: int = 30,
        sample_rate: int = 16000,
    ) -> None:
        self.examples = examples
        self.seed = seed
        self.total_frames = total_frames
        self.time_steps = total_frames // video_frames_per_step
        self.action_dimension = action_dimension
        self.image_size = image_size
        self.video_rate = video_rate
        self.sample_rate = sample_rate
        self.wall_half_width = image_size // 4
        self.object_radius = image_size // 14
        self.hidden_offset = self.wall_half_width - self.object_radius
        self.maximum_offset = image_size // 2 - self.object_radius

    def __len__(self) -> int:
        return self.examples

    def object_offsets(self, seconds: Tensor, emergence_seconds: float, start_offset: float, direction: int) -> Tensor:
        speed = (self.hidden_offset - start_offset) / emergence_seconds
        return direction * (start_offset + speed * seconds).clamp(max=self.maximum_offset)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        generator = torch.Generator().manual_seed(self.seed * 1_000_003 + index)
        side = int(torch.randint(0, 2, (1,), generator=generator))
        direction = 1 if side else -1
        emergence_seconds = 0.6 + 0.3 * float(torch.rand(1, generator=generator))
        step_interval = 0.15 + 0.1 * float(torch.rand(1, generator=generator))
        timbre = float(torch.rand(1, generator=generator))
        start_offset = self.hidden_offset * (0.25 + 0.25 * float(torch.rand(1, generator=generator)))
        color = (
            torch.rand(3, generator=generator) * 0.6
            + torch.tensor([0.4, 0.1, 0.1])[torch.randperm(3, generator=generator)]
        )

        size = self.image_size
        center = size // 2
        frame_seconds = torch.arange(self.total_frames) / self.video_rate
        offsets = self.object_offsets(frame_seconds, emergence_seconds, start_offset, direction)
        columns = repeat(torch.arange(size).float(), "column -> row column", row=size)
        rows = repeat(torch.arange(size).float(), "row -> row column", column=size)
        object_masks = (
            (
                rearrange(columns, "row column -> 1 row column") - rearrange(center + offsets, "frame -> frame 1 1")
            ).square()
            + (rearrange(rows, "row column -> 1 row column") - center).square()
            <= self.object_radius**2
        ).float()
        wall = (((columns - center).abs() <= self.wall_half_width) & ((rows - center).abs() <= size * 0.32)).float()
        background = render_background(size, generator)
        object_layer = rearrange(color, "channel -> 1 channel 1 1")
        wall_color = torch.tensor([0.16, 0.15, 0.17])[:, None, None]
        frames = background[None] * (1 - object_masks[:, None]) + object_layer * object_masks[:, None]
        frames = frames * (1 - wall[None, None]) + wall_color[None] * wall[None, None]

        samples = int(self.total_frames / self.video_rate * self.sample_rate)
        waveform = 0.003 * torch.randn(2, samples, generator=generator)
        step_start = float(torch.rand(1, generator=generator)) * step_interval
        while step_start < samples / self.sample_rate:
            offset = self.object_offsets(torch.tensor([step_start]), emergence_seconds, start_offset, direction)
            pan = float((offset[0] / self.hidden_offset).clamp(-1.0, 1.0))
            angle = (pan + 1.0) * math.pi / 4
            sound = footstep(self.sample_rate, timbre, generator)
            start = int(step_start * self.sample_rate)
            end = min(samples, start + sound.shape[0])
            waveform[0, start:end] += math.cos(angle) * 0.4 * sound[: end - start]
            waveform[1, start:end] += math.sin(angle) * 0.4 * sound[: end - start]
            step_start += step_interval * (0.9 + 0.2 * float(torch.rand(1, generator=generator)))

        return {
            "frames": frames,
            "waveform": waveform.clamp(-1.0, 1.0),
            "actions": torch.zeros(self.time_steps, self.action_dimension),
            "action_present": torch.tensor(0, dtype=torch.int64),
            "example_seed": torch.tensor(index, dtype=torch.int64),
            "side": torch.tensor(side, dtype=torch.int64),
            "emergence_seconds": torch.tensor(emergence_seconds),
        }
