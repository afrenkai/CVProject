from collections import Counter
from pathlib import Path
import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pydantic import Field, NonNegativeInt, PositiveFloat, PositiveInt
from torch.utils.data import DataLoader
from architecture.pretraining.joint_encoder import JointEncoder
from evaluation.evaluation_types import PrefixCheckConfig
from evaluation.prefix_leakage import assert_prefix_independence
from train_datasets.calibration import calibration_windows
from train_datasets.corpus_dataset import CorpusWindows
from train_datasets.manifests import window_corpus
from train_datasets.records import read_windows, write_yaml
from training.pretraining.configuration import FrozenInputConfig
from training.pretraining.normalization import ActionStatistics, FeatureStatistics
from training.runtime import cuda_device


class NormalizationConfig(FrozenInputConfig):
    windows: Path
    output: Path
    sample_windows: PositiveInt = 1024
    workers: NonNegativeInt = 2
    seed: NonNegativeInt = 67
    minimum_standard_deviation: PositiveFloat = 0.000001
    prefix_checks: PrefixCheckConfig = Field(default_factory=PrefixCheckConfig)


@torch.no_grad()
def run_normalization(config: NormalizationConfig) -> None:
    config.output.mkdir(parents=True, exist_ok=True)
    if any((config.output / name).exists() for name in ("normalizer.pt", "action_normalizer.yaml", "completed.yaml")):
        raise FileExistsError("Choose a fresh normalization output; never replace scales beneath a resumed model")
    windows = read_windows(config.windows, "train")
    selected = calibration_windows(windows, config.sample_windows, config.seed)
    write_yaml(config.output / "selected_windows.yaml", [window.model_dump(mode="json") for window in selected])
    device = cuda_device(config.seed)
    encoder = JointEncoder(config.encoder, device)
    dataset = CorpusWindows(selected, config.data)
    features = FeatureStatistics(config.minimum_standard_deviation)
    actions = ActionStatistics(len(config.data.control_layout.axis_names))
    guards: dict[str, object] = {}
    for index, batch in enumerate(DataLoader(dataset, batch_size=1, num_workers=config.workers)):
        frames, waveform = batch["frames"].to(device), batch["waveform"].to(device)
        corpus = window_corpus(selected[index])
        if corpus not in guards:
            guards[corpus] = {
                "source": selected[index].source.source_id,
                "interventions": assert_prefix_independence(
                    encoder,
                    frames,
                    waveform,
                    config.encoder,
                    config.prefix_checks,
                    config.seed,
                ),
            }
        video, audio = encoder.encode(frames, waveform)
        assert video.dtype == audio.dtype == torch.float32
        features.observe(video, audio)
        actions.observe(batch["actions"], batch["action_present"])
        print(f"normalizer_train_window {index + 1}/{len(selected)} {corpus}", flush=True)
    normalizer = features.normalizer()
    pending = config.output / "normalizer.pending"
    torch.save(normalizer.state_dict(), pending)
    pending.replace(config.output / "normalizer.pt")
    write_yaml(
        config.output / "action_normalizer.yaml",
        {
            "fit_split": "train",
            "fit_windows": len(selected),
            "axis_rows": actions.rows,
            "axis_scales": actions.scales(),
            "layout": config.data.control_layout.model_dump(),
        },
    )
    report = {
        "completed": True,
        "fit_split": "train",
        "fit_windows": len(selected),
        "requested_fit_windows": config.sample_windows,
        "training_candidate_windows": len(windows),
        "unique_training_windows": len({(window.source.source_id, window.start_seconds) for window in windows}),
        "by_corpus": dict(Counter(window_corpus(window) for window in selected)),
        "selection": "Seeded unique TRAIN windows; at least one per corpus, uniform remaining candidates",
        "encoder_precision": "float32",
        "minimum_standard_deviation": config.minimum_standard_deviation,
        "prefix_guards": guards,
        "axis_rows": actions.rows,
        "axis_scales": actions.scales(),
        "validation_test_cs_r6_fit_windows": 0,
        "fresh_pretraining_initialization_required": True,
    }
    write_yaml(config.output / "completed.yaml", report)
    print(OmegaConf.to_yaml(OmegaConf.create(report)), flush=True)


@hydra.main(config_path="../../configs", config_name="fit_normalizer", version_base=None)
def main(config: DictConfig) -> None:
    settings = instantiate(config)
    assert isinstance(settings, NormalizationConfig)
    run_normalization(settings)


if __name__ == "__main__":
    main()
