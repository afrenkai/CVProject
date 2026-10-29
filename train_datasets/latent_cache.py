import time
from pathlib import Path
import torch
from huggingface_hub import snapshot_download
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig
from architecture.pretraining.encoder_types import FrozenEncoderConfig
from architecture.pretraining.joint_encoder import JointEncoder


def build_encoder_config(context_frames: int, audio_model: str, bfloat16: bool = False) -> FrozenEncoderConfig:
    audio_path = Path(snapshot_download(audio_model))
    audio_feature_dim: int = AutoConfig.from_pretrained(
        str(audio_path), trust_remote_code=True, local_files_only=True
    ).encoder_kwargs["embed_dim"]
    return FrozenEncoderConfig(
        video_model="vit_base_patch16_dinov3.lvd1689m",
        video_feature_dimension=768,
        video_layers=(3, 6, 9, 12),
        image_size=112,
        video_rate=30,
        video_frames_per_step=3,
        sample_rate=16000,
        context_frames=context_frames,
        audio_model_directory=audio_path,
        audio_feature_dimension=audio_feature_dim,
        audio_native_frame_rate=25,
        audio_frame_repeat=2,
        audio_tokens_per_step=5,
        bfloat16=bfloat16,
    )


def encode_dataset(
    dataset: Dataset,
    encoder_cfg: FrozenEncoderConfig,
    device: str,
    batch_size: int,
    workers: int,
    label_keys: tuple[str, ...] = (),
) -> dict[str, Tensor | dict]:
    encoder = JointEncoder(encoder_cfg, torch.device(device))
    context_samples = encoder_cfg.context_frames * encoder_cfg.sample_rate // encoder_cfg.video_rate
    chunks: dict[str, list[Tensor]] = {}
    started = time.perf_counter()
    done = 0
    for batch in DataLoader(dataset, batch_size=batch_size, num_workers=workers):
        waveform = batch["waveform"]
        with torch.no_grad():
            video, audio = encoder.encode(batch["frames"].to(device), waveform.to(device))
        chunks.setdefault("video", []).append(video.half().cpu())
        chunks.setdefault("audio", []).append(audio.half().cpu())
        chunks.setdefault("target_rms", []).append(waveform[:, :, context_samples:].square().mean(dim=(1, 2)).sqrt())
        for key in label_keys:
            chunks.setdefault(key, []).append(batch[key])
        done += waveform.shape[0]
        elapsed = time.perf_counter() - started
        print(f"encoded {done}/{len(dataset)} windows, {elapsed / done:.2f}s per window", flush=True)
    return {"encoder": encoder_cfg.model_dump(mode="json")} | {key: torch.cat(values) for key, values in chunks.items()}
