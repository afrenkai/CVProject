from pathlib import Path
from omegaconf import DictConfig, OmegaConf


def load_settings(config: Path) -> DictConfig:
    settings = OmegaConf.load(config)
    parents = [load_settings(config.parent / parent) for parent in settings.pop("inherit", [])]
    return OmegaConf.merge(*parents, settings)


def load_run_settings(config: Path, overrides: list[str]) -> dict[str, object]:
    merged = OmegaConf.merge(load_settings(config), OmegaConf.from_dotlist(overrides))
    return OmegaConf.to_container(merged, resolve=True)
