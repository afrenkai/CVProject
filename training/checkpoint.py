from pathlib import Path
from typing import cast
import torch
from training.types import Checkpoint, ModelModules


def save_checkpoint(path: Path, modules: ModelModules) -> Path:
    checkpoint = {name: dict(module.state_dict()) for name, module in modules.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".partial")
    torch.save(checkpoint, temporary_path)
    temporary_path.replace(path)
    return path


def load_checkpoint(path: Path, modules: ModelModules) -> None:
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(loaded, dict):
        raise TypeError(f"Invalid checkpoint: {path}")
    if set(loaded) != set(modules):
        raise ValueError(f"Checkpoint keys must be {sorted(modules)}")
    checkpoint = cast(Checkpoint, loaded)
    for name, module in modules.items():
        module.load_state_dict(checkpoint[name])
