import time
from pathlib import Path
from typing import ClassVar
import torch
from safetensors.torch import load_file
from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.distributed.parallel_state import get_ring_ctx, get_sp_group, get_ulysses_ctx
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import ComposedPipelineBase
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from torch import nn
from torch.distributed import ProcessGroup
from data.control_types import ControlLayout
from inference.inference_types import SpeakUpPipelineConfig, SpeakUpSamplingParams
from inference.stages import (
    EncodeCandidateControls,
    PrepareFlowTimesteps,
    PrepareJointAVNoise,
    PrepareObservedContext,
    ReturnLatents,
    SpeakUpDenoisingStage,
    ValidateForecast,
    denoise_rows,
)
from models.compilation import compile_blocks, set_recompile_limit
from models.encoder_types import FrozenEncoderConfig
from models.joint_encoder import JointEncoder, build_encoder_config
from models.transformers import AdaptiveAttentionBlock
from models.world_model import AudioVisualFlowPredictor
from training.objective import WorldModelObjective, restore_objective
from training.training_types import PretrainConfig


def load_bundle(bundle: Path, config: SpeakUpPipelineConfig, device: torch.device) -> dict[str, nn.Module]:
    settings = PretrainConfig.model_validate_json((bundle / "pretrain_config.json").read_text())
    settings = settings.model_copy(
        update={"model": settings.model.model_copy(update={"attention_backend": config.attention_backend})}
    )
    encoder_config = build_encoder_config(settings.encoder, bfloat16=True)
    objective = restore_objective(settings, encoder_config, load_file(bundle / "model.safetensors"), device)
    if get_ring_ctx()[0] > 1:
        raise ValueError("SpeakUpPipeline supports Ulysses sequence parallelism only; set --ring-degree 1")
    if get_ulysses_ctx()[0] > 1:
        enable_sequence_parallel(objective.predictor, get_sp_group().ulysses_group)
    if config.compile:
        set_recompile_limit(config.recompile_limit)
        compile_blocks(objective.predictor)
        warm_up(objective, encoder_config, settings.encoder.total_frames, config)
    return {"objective": objective, "encoder": JointEncoder(encoder_config, device).eval()}


def enable_sequence_parallel(predictor: AudioVisualFlowPredictor, group: ProcessGroup) -> None:
    predictor.sequence_group = group
    predictor.diffusion_head.sequence_group = group
    for module in predictor.modules():
        if isinstance(module, AdaptiveAttentionBlock):
            module.self_attention.sequence_group = group


def warm_up(
    objective: WorldModelObjective, encoder: FrozenEncoderConfig, total_frames: int, config: SpeakUpPipelineConfig
) -> None:
    device = objective.normalizer.video.mean.device
    patches = (encoder.image_size // 16) ** 2
    context_steps = objective.context_steps
    target_steps = total_frames // encoder.video_frames_per_step - context_steps
    context_video = torch.zeros(1, context_steps, patches, encoder.video_latent_dimension, device=device)
    context_audio = torch.zeros(1, context_steps, encoder.audio_latent_dimension, device=device)
    for bucket in config.row_buckets:
        start = time.perf_counter()
        noise_video = torch.randn(bucket, target_steps, patches, encoder.video_latent_dimension, device=device)
        noise_audio = torch.randn(bucket, target_steps, encoder.audio_latent_dimension, device=device)
        actions = objective.encode_actions(
            torch.zeros(1, context_steps + target_steps, ControlLayout().dimension, device=device),
            torch.zeros(1, dtype=torch.int64, device=device),
        ).expand(bucket, -1, -1)
        grid = objective.transport.time_grid(1, noise_video)
        denoise_rows(
            objective, context_video, context_audio, noise_video, noise_audio, actions, grid, bucket, config.bfloat16
        )
        torch.cuda.synchronize()
        print(f"Warmed up {bucket} rows in {time.perf_counter() - start:.1f}s", flush=True)


class SpeakUpPipeline(ComposedPipelineBase):
    pipeline_name = "SpeakUpPipeline"
    pipeline_config_cls = SpeakUpPipelineConfig
    sampling_params_cls = SpeakUpSamplingParams
    _required_config_modules: ClassVar[list[str]] = []

    def validate_disagg_role(self, role: RoleType) -> None:
        if role != RoleType.MONOLITHIC:
            raise ValueError("SpeakUpPipeline runs monolithic replicas; scale with --dp-size")

    def load_modules(
        self, server_args: ServerArgs, loaded_modules: dict[str, nn.Module] | None = None
    ) -> dict[str, nn.Module]:
        if loaded_modules is not None:
            return loaded_modules
        config: SpeakUpPipelineConfig = server_args.pipeline_config
        device = torch.device("cuda", torch.cuda.current_device())
        return load_bundle(Path(self.model_path), config, device)

    def create_pipeline_stages(self, server_args: ServerArgs) -> None:
        objective = self.get_module("objective")
        encoder: JointEncoder = self.get_module("encoder")
        bundle = PretrainConfig.model_validate_json((Path(self.model_path) / "pretrain_config.json").read_text())
        self.add_stage(
            ValidateForecast(objective, encoder.settings, bundle.encoder.total_frames, ControlLayout().dimension),
            "speakup_validate",
        )
        self.add_stage(PrepareObservedContext(objective, encoder), "speakup_context")
        self.add_stage(EncodeCandidateControls(objective), "speakup_controls")
        self.add_stage(PrepareJointAVNoise(objective), "speakup_noise")
        self.add_stage(PrepareFlowTimesteps(objective), "speakup_timesteps")
        self.add_stage(SpeakUpDenoisingStage(objective), "speakup_denoise")
        self.add_stage(ReturnLatents(), "speakup_return")


EntryClass = SpeakUpPipeline
