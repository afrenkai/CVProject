from pathlib import Path
import fire
import pandas as pd
import torch
from train_datasets.control_types import ControlLayout
from train_datasets.latent_cache import build_encoder_config, encode_dataset
from train_datasets.opencs2_events import local_path
from train_datasets.simple_av import AVClip, SimpleAVDataset


def main(
    events: str,
    split: str,
    output: str,
    directory: str = "/scratch/afrenk/speakup/opencs2",
    repository: str = "blanchon/opencs2_dataset",
    batch_size: int = 16,
    workers: int = 12,
    device: str = "cuda",
    bfloat16: bool = True,
) -> None:
    frame = pd.read_parquet(events)
    frame = frame[frame["split"] == split]
    videos = {video: local_path(repository, video, Path(directory)) for video in frame["video"].unique()}
    print(f"{split}: {len(frame)} events from {len(videos)} videos", flush=True)
    clips = [AVClip(videos[row.video], row.start_seconds, row.side) for row in frame.itertuples(index=False)]
    encoder_cfg = build_encoder_config(context_frames=36, audio_model="mispeech/dasheng-base", bfloat16=bfloat16)
    dataset = SimpleAVDataset(
        clips,
        total_frames=60,
        video_frames_per_step=encoder_cfg.video_frames_per_step,
        action_dimension=ControlLayout().dimension,
        image_size=encoder_cfg.image_size,
        video_rate=encoder_cfg.video_rate,
        sample_rate=encoder_cfg.sample_rate,
    )
    cache = encode_dataset(dataset, encoder_cfg, device, batch_size, workers, ("side",))
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, output)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    fire.Fire(main)
