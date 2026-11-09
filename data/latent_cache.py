import argparse
from bisect import bisect_right
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset
from configs.loading import load_run_settings
from data.clips import AVDataset, build_dataset, build_evaluation_datasets
from data.control_types import ControlLayout
from models.joint_encoder import JointEncoder, build_encoder_config
from training.training_types import PretrainConfig


class CachedClipDataset(Dataset):
    def __init__(self, directory: Path, clips: int | None = None) -> None:
        self.shards = sorted(directory.glob("shard_*.safetensors"))
        assert self.shards, f"No latent shards in {directory}"
        self.lengths = [self.rows(shard) for shard in self.shards]
        self.starts = [sum(self.lengths[:index]) for index in range(len(self.lengths))]
        assert [
            int(shard.stem.removeprefix("shard_")) for shard in self.shards
        ] == self.starts, f"Latent shards in {directory} are not (C) contiguous"
        self.total = sum(self.lengths) if clips is None else min(clips, sum(self.lengths))
        self.handles: dict[int, safe_open] = {}

    def rows(self, shard: Path) -> int:
        with safe_open(str(shard), "pt") as handle:
            return handle.get_slice("video").get_shape()[0]

    def __len__(self) -> int:
        return self.total

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """
        lookup using bisect is O(log n)
        regardless, this is still a "grid" of shardsxidx which we open via safetensors' safe open

        """
        shard = bisect_right(self.starts, index) - 1
        row = index - self.starts[shard]
        if shard not in self.handles:
            self.handles[shard] = safe_open(str(self.shards[shard]), "pt")
        handle = self.handles[shard]
        return {
            "video": handle.get_slice("video")[row],
            "audio": handle.get_slice("audio")[row],
            "actions": handle.get_slice("actions")[row].float(),
            "action_present": handle.get_slice("action_present")[row].long(),
        }


@torch.no_grad()
def write_shards(
    dataset: AVDataset,
    encoder: JointEncoder,
    directory: Path,
    start: int,
    count: int,
    shard_clips: int,
    batch_size: int,
    workers: int,
    device: torch.device,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for shard_start in range(start, min(start + count, len(dataset)), shard_clips):
        path = directory / f"shard_{shard_start:08d}.safetensors"
        if path.exists():
            continue
        indices = range(shard_start, min(shard_start + shard_clips, start + count, len(dataset)))
        parts: dict[str, list[Tensor]] = {"video": [], "audio": [], "actions": [], "action_present": []}
        for batch in DataLoader(Subset(dataset, indices), batch_size=batch_size, num_workers=workers):
            video, audio = encoder.encode(batch["frames"].to(device), batch["waveform"].to(device))
            parts["video"].append(video.to(torch.bfloat16).cpu())
            parts["audio"].append(audio.to(torch.bfloat16).cpu())
            parts["actions"].append(batch["actions"].to(torch.float16))
            parts["action_present"].append(batch["action_present"].to(torch.int8))
        partial = path.with_suffix(".part")
        save_file(
            {name: torch.cat(values).contiguous() for name, values in parts.items()},
            str(partial),
        )
        partial.rename(path)
        print(f"wrote {path} ({len(indices)} clips)", flush=True)


def main(
    config_path: Path,
    split: str,
    start: int,
    count: int,
    overrides: list[str],
) -> None:
    config = PretrainConfig.model_validate(load_run_settings(config_path, overrides))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder_config = build_encoder_config(config.encoder, bfloat16=config.trainer.precision == "bf16")
    encoder = JointEncoder(encoder_config, device).eval()
    output = Path(config.data.latent_cache)
    layout = ControlLayout()
    settings = {"shard_clips": 1024, "batch_size": 32, "workers": config.data.workers, "device": device}
    if split == "train":
        dataset = build_dataset(config.data, encoder_config, config.encoder.total_frames, layout)
        write_shards(
            dataset,
            encoder,
            output / "train",
            start,
            count,
            **settings,
        )
        return
    for name, dataset in build_evaluation_datasets(
        config.data, split, encoder_config, config.encoder.total_frames, layout
    ).items():
        write_shards(
            dataset,
            encoder,
            output / split / name,
            0,
            len(dataset),
            **settings,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--split", choices=["train", "validation", "test"], default="train")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=1 << 40)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    main(
        args.config,
        args.split,
        args.start,
        args.count,
        args.overrides,
    )
