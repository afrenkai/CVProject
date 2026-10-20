import hashlib
from pathlib import Path
from typing import cast

import torch
from omegaconf import DictConfig, OmegaConf

from training.checkpoint import load_checkpoint, save_checkpoint
from training.pretraining.models import build_pretraining_models
from training.pretraining.types import PretrainingModels
from training.types import Checkpoint


def pretraining_checkpoint_hash(directory: Path) -> str:
    fingerprint = hashlib.sha256()
    for filename in ("model.pt", "config.yaml"):
        with (directory / filename).open("rb") as checkpoint_stream:
            fingerprint.update(hashlib.file_digest(checkpoint_stream, "sha256").digest())
    return fingerprint.hexdigest()


def frozen_feature_modules_match(reference: Path, checkpoint: Path) -> bool:
    reference_state = cast(Checkpoint, torch.load(reference / "model.pt", map_location="cpu", weights_only=True))
    checkpoint_state = cast(Checkpoint, torch.load(checkpoint / "model.pt", map_location="cpu", weights_only=True))
    for module_name in ("video_encoder", "audio_encoder", "normalizer"):
        if reference_state[module_name].keys() != checkpoint_state[module_name].keys():
            return False
        for tensor_name, reference_tensor in reference_state[module_name].items():
            if not torch.equal(reference_tensor, checkpoint_state[module_name][tensor_name]):
                return False
    return True


def save_pretraining_checkpoint(directory: Path, models: PretrainingModels, config: DictConfig) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, directory / "config.yaml", resolve=True)
    return save_checkpoint(directory / "model.pt", models.pretraining_checkpoint_modules())


def load_pretraining_checkpoint(directory: Path, device: torch.device) -> tuple[PretrainingModels, DictConfig]:
    config = OmegaConf.load(directory / "config.yaml")
    if not isinstance(config, DictConfig):
        raise TypeError("Pretraining configuration must be a mapping")
    models = build_pretraining_models(config, device, pretrained_video=False)
    modules = models.pretraining_checkpoint_modules()
    load_checkpoint(directory / "model.pt", modules)
    for module in modules.values():
        module.requires_grad_(False)
        module.eval()
    return models, config
