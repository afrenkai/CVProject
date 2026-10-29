from pathlib import Path
import fire
import torch
from train_datasets.control_types import ControlLayout
from train_datasets.latent_cache import build_encoder_config, encode_dataset
from train_datasets.synthetic_occlusion import SyntheticOcclusionDataset


def main(
    examples: int,
    seed: int,
    output: str,
    audio_model: str = "mispeech/dasheng-base",
    batch_size: int = 4,
    device: str = "cpu",
    workers: int = 0,
) -> None:
    encoder_cfg = build_encoder_config(context_frames=12, audio_model=audio_model)
    dataset = SyntheticOcclusionDataset(
        examples,
        seed,
        total_frames=42,
        video_frames_per_step=encoder_cfg.video_frames_per_step,
        action_dimension=ControlLayout().dimension,
        image_size=encoder_cfg.image_size,
        video_rate=encoder_cfg.video_rate,
        sample_rate=encoder_cfg.sample_rate,
    )
    cache = encode_dataset(dataset, encoder_cfg, device, batch_size, workers, ("side", "emergence_seconds"))
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, output)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    fire.Fire(main)
