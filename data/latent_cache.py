import argparse
import json
from bisect import bisect_right
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset
from configs.loading import load_run_settings
from data.control_types import ControlLayout
from data.windows import PoolWindowDataset, build_dataset, build_validation_datasets
from models.joint_encoder import JointEncoder, build_encoder_config
from training.training_types import PretrainConfig


class CachedWindowDataset(Dataset):
    def __init__(self, directory: Path, windows: int | None = None) -> None:
        self.shards = sorted(directory.glob("shard_*.safetensors"))
        assert self.shards, f"No latent shards in {directory}"
        self.lengths = [int(json.loads(self.metadata(shard)["windows"])) for shard in self.shards]
        self.starts = [sum(self.lengths[:index]) for index in range(len(self.lengths))]
        self.total = sum(self.lengths) if windows is None else min(windows, sum(self.lengths))
        self.handles: dict[int, object] = {}
        self.datasets = json.loads(self.metadata(self.shards[0])["datasets"])

    def metadata(self, shard: Path) -> dict[str, str]:
        with safe_open(str(shard), "pt") as handle:
            return handle.metadata()

    def __len__(self) -> int:
        return self.total

    def __getitem__(self, index: int) -> dict[str, Tensor]:
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
    dataset: PoolWindowDataset,
    encoder: JointEncoder,
    directory: Path,
    start: int,
    count: int,
    shard_windows: int,
    batch_size: int,
    workers: int,
    device: torch.device,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    names = sorted(set(dataset.datasets.tolist()))
    for shard_start in range(start, min(start + count, len(dataset)), shard_windows):
        path = directory / f"shard_{shard_start:08d}.safetensors"
        if path.exists():
            continue
        indices = range(shard_start, min(shard_start + shard_windows, start + count, len(dataset)))
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
            metadata={"windows": json.dumps(len(indices)), "datasets": json.dumps(names)},
        )
        partial.rename(path)
        print(f"wrote {path} ({len(indices)} windows)", flush=True)


def main(config_path: Path, output: Path, split: str, start: int, count: int, overrides: list[str]) -> None:
    config = PretrainConfig.model_validate(load_run_settings(config_path, overrides))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder_config = build_encoder_config(config.encoder, bfloat16=config.trainer.precision == "bf16")
    encoder = JointEncoder(encoder_config, device).eval()
    layout = ControlLayout()
    settings = {"shard_windows": 1024, "batch_size": 32, "workers": config.data.workers, "device": device}
    if split == "train":
        dataset = build_dataset(config.data, encoder_config, config.encoder.total_frames, layout)
        write_shards(dataset, encoder, output / "train", start, count, **settings)
        return
    for name, dataset in build_validation_datasets(
        config.data, encoder_config, config.encoder.total_frames, layout
    ).items():
        write_shards(dataset, encoder, output / "validation" / name, 0, len(dataset), **settings)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--split", choices=["train", "validation"], default="train")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=1 << 40)
    parser.add_argument("overrides", nargs="*")
    command = parser.parse_args()
    main(command.config, command.output, command.split, command.start, command.count, command.overrides)
