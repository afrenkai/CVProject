from pathlib import Path
import fire
import torch
from datasets import load_dataset
from train_datasets.control_types import ControlLayout
from train_datasets.latent_cache import build_encoder_config, encode_dataset
from train_datasets.simple_av import HFAVDataset, scan_hf_windows


def main(
    first_actor: int,
    last_actor: int,
    videos_per_actor: int,
    output: str,
    hf_dataset: str = "mteb/RAVDESS_AV",
    audio_model: str = "mispeech/dasheng-base",
    batch_size: int = 4,
    device: str = "cpu",
    workers: int = 0,
) -> None:
    video_rate = 30
    total_frames = 30
    encoder_cfg = build_encoder_config(context_frames=6, audio_model=audio_model)

    hf_ds = load_dataset(hf_dataset, split="test")
    row_actors = [int(video_id.removesuffix(".mp4").split("-")[6]) for video_id in hf_ds["video_id"]]
    actor_rows: dict[int, list[int]] = {}
    for row, actor in enumerate(row_actors):
        if first_actor <= actor <= last_actor:
            actor_rows.setdefault(actor, []).append(row)
    rows = [
        actor_row
        for actor in sorted(actor_rows)
        for actor_row in actor_rows[actor][:: max(1, len(actor_rows[actor]) // videos_per_actor)][:videos_per_actor]
    ]
    subset = hf_ds.select(rows)
    windows = scan_hf_windows(subset, total_frames, video_rate, stride_seconds=0.5)
    dataset = HFAVDataset(
        subset,
        windows,
        total_frames,
        encoder_cfg.video_frames_per_step,
        ControlLayout().dimension,
        encoder_cfg.image_size,
        video_rate,
    )
    print(f"{len(rows)} videos from actors {first_actor}-{last_actor}, {len(windows)} windows", flush=True)

    cache = encode_dataset(dataset, encoder_cfg, device, batch_size, workers)

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        cache
        | {
            "actor": torch.tensor([row_actors[rows[subset_row]] for subset_row, start in windows]),
            "row": torch.tensor([rows[subset_row] for subset_row, start in windows]),
        },
        output,
    )
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    fire.Fire(main)
