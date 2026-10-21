from pathlib import Path
from typing import cast

import torch
from omegaconf import DictConfig, OmegaConf

from training.checkpoint import load_checkpoint, save_checkpoint
from training.pretraining.models import build_pretraining_models
from training.pretraining.types import PretrainingModels


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
