import json
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from sglang.multimodal_gen.configs.pipeline_configs.base import PipelineConfig
from sglang.multimodal_gen.configs.sample.action import ActionSamplingParams
from sglang.multimodal_gen.configs.task_type import ModelTaskType
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from data.control_types import ControlLayout
from models.joint_encoder import build_encoder_config
from training.training_types import PretrainConfig

Observation = dict[str, np.ndarray | list | int | float | str]


@dataclass
class SpeakUpPipelineConfig(PipelineConfig):
    task_type: ModelTaskType = ModelTaskType.VLA_ACTION
    should_use_guidance: bool = False
    attention_backend: str = "flex"
    compile: bool = True
    recompile_limit: int = 512
    bfloat16: bool = True
    default_num_inference_steps: int = 16
    action_horizon: int = 28
    action_dim: int = 265
    output_action_dim: int = 265
    image_keys: tuple[str, ...] = ()
    row_buckets: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)
    max_rows_per_request: int = 1024

    def action_metadata(self, server_args: ServerArgs) -> dict[str, object]:
        bundle = Path(server_args.model_path)
        config = PretrainConfig.model_validate(json.loads((bundle / "pretrain_config.json").read_text()))
        encoder = build_encoder_config(config.encoder, bfloat16=self.bfloat16)
        total_steps = config.encoder.total_frames // encoder.video_frames_per_step
        context_steps = encoder.context_frames // encoder.video_frames_per_step
        return {
            "model": server_args.served_model_name,
            "weights": json.loads((bundle / "model_index.json").read_text())["weights"],
            "latent_space": "encoder",
            "context_steps": context_steps,
            "target_steps": total_steps - context_steps,
            "patches": (encoder.image_size // 16) ** 2,
            "video_latent_dimension": encoder.video_latent_dimension,
            "audio_latent_dimension": encoder.audio_latent_dimension,
            "control_dimension": ControlLayout().dimension,
            "context_frames": encoder.context_frames,
            "video_rate": encoder.video_rate,
            "video_frames_per_step": encoder.video_frames_per_step,
            "image_size": encoder.image_size,
            "waveform_channels": encoder.waveform_channels,
            "sample_rate": encoder.loading_sample_rate,
            "context_samples": round(encoder.context_frames / encoder.video_rate * encoder.loading_sample_rate),
            "default_num_inference_steps": self.default_num_inference_steps,
            "max_rows_per_request": self.max_rows_per_request,
        }


@dataclass
class SpeakUpSamplingParams(ActionSamplingParams):
    num_inference_steps: int = 16
    action_horizon: int = 28
    action_dim: int = 265
    output_format: str = "list"
    return_timing: bool = True
    observation: Observation | None = field(default=None, metadata={"batch_sig_exclude": True})

    def build_request_extra(self) -> dict[str, dict[str, object]]:
        extra = super().build_request_extra()
        extra["speakup"] = {
            "observation": dict(self.observation or {}),
            "output_format": self.output_format,
            "return_timing": self.return_timing,
        }
        return extra

    def _validate(self) -> None:
        super()._validate()
        if self.output_format not in ("list", "numpy"):
            raise ValueError("output_format must be 'list' or 'numpy'")
        observation = self.observation or {}
        latent_keys = {"context_video", "context_audio"} & observation.keys()
        raw_keys = {"frames", "waveform"} & observation.keys()
        if observation and len(latent_keys) + len(raw_keys) != 2:
            raise ValueError("observation needs context_video and context_audio, or frames and waveform")
        if observation and latent_keys and raw_keys:
            raise ValueError("observation must not mix latent and raw context")
        samples = observation.get("samples_per_candidate", 1)
        if isinstance(samples, bool) or not isinstance(samples, int) or samples <= 0:
            raise ValueError("samples_per_candidate must be a positive int")
