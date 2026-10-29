from pathlib import Path
from typing import cast
import torch
from omegaconf import DictConfig, OmegaConf
from training.pretraining.models import build_pretraining_models
from training.pretraining.pretraining_types import PretrainingModels
from training.training_types import Checkpoint, ModelModules


def save_checkpoint(path: Path, modules: ModelModules) -> Path:
    checkpoint = {name: dict(module.state_dict()) for name, module in modules.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".partial")
    torch.save(checkpoint, temporary_path)
    temporary_path.replace(path)
    return path


def load_checkpoint(path: Path, modules: ModelModules) -> None:
    checkpoint = cast(Checkpoint, torch.load(path, map_location="cpu", weights_only=True))
    for name, module in modules.items():
        module.load_state_dict(checkpoint[name])


def save_pretraining_checkpoint(directory: Path, models: PretrainingModels, config: DictConfig) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, directory / "config.yaml", resolve=True)
    return save_checkpoint(directory / "model.pt", models.pretraining_checkpoint_modules())


def load_pretraining_checkpoint(directory: Path, device: torch.device) -> tuple[PretrainingModels, DictConfig]:
    config = cast(DictConfig, OmegaConf.load(directory / "config.yaml"))
    models = build_pretraining_models(config, device, pretrained_video=False)
    modules = models.pretraining_checkpoint_modules()
    load_checkpoint(directory / "model.pt", modules)
    for module in modules.values():
        module.requires_grad_(False)
        module.eval()
    return models, config
