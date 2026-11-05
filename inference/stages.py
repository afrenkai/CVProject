import time
import numpy as np
import torch
from einops import rearrange, repeat
from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from torch import Tensor
from inference.inference_types import Observation, SpeakUpPipelineConfig
from models.encoder_types import FrozenEncoderConfig
from models.guidance import GuidedPredictor
from models.joint_encoder import JointEncoder
from training.objective import WorldModelObjective
from transport.solvers import solve_flow_map


def speakup_state(batch: Req) -> dict[str, object]:
    return batch.extra["speakup"]


def record_timing(batch: Req, name: str, start: float) -> None:
    timings = speakup_state(batch).setdefault("timings", {})
    timings[name] = (time.perf_counter() - start) * 1000


def float_array(value: object, name: str, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {list(shape)}, got {list(array.shape)}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def request_seed(batch: Req) -> int:
    seed = batch.seed
    return seed[0] if isinstance(seed, list) else seed


class ValidateForecast(PipelineStage):
    def __init__(
        self, objective: WorldModelObjective, encoder_config: FrozenEncoderConfig, total_frames: int, control: int
    ) -> None:
        super().__init__()
        self.objective = objective
        self.encoder_config = encoder_config
        self.total_steps = total_frames // encoder_config.video_frames_per_step
        self.control = control

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        start = time.perf_counter()
        state = speakup_state(batch)
        observation: Observation = state["observation"]
        config: SpeakUpPipelineConfig = server_args.pipeline_config
        settings = self.encoder_config
        context_steps = self.objective.context_steps
        video_shape = (context_steps, (settings.image_size // 16) ** 2, settings.video_latent_dimension)
        audio_shape = (context_steps, settings.audio_latent_dimension)
        match observation:
            case {"context_video": video, "context_audio": audio}:
                state["context_video"] = float_array(video, "context_video", video_shape)
                state["context_audio"] = float_array(audio, "context_audio", audio_shape)
            case {"frames": frames, "waveform": waveform}:
                frames = np.asarray(frames, dtype=np.uint8)
                if frames.ndim != 4 or frames.shape[0] != settings.context_frames or frames.shape[-1] != 3:
                    raise ValueError(f"frames must have shape [{settings.context_frames}, height, width, 3]")
                samples = round(settings.context_frames / settings.video_rate * settings.loading_sample_rate)
                state["frames"] = frames
                state["waveform"] = float_array(waveform, "waveform", (settings.waveform_channels, samples))
            case {} if not observation and batch.is_warmup:
                state["context_video"] = np.zeros(video_shape, np.float32)
                state["context_audio"] = np.zeros(audio_shape, np.float32)
            case unknown:
                raise ValueError(f"observation needs context latents or frames and waveform, got {sorted(unknown)}")
        match observation.get("controls"):
            case None:
                controls = np.zeros((1, self.total_steps, self.control), dtype=np.float32)
                state["action_present"] = np.zeros(1, dtype=np.int64)
            case given:
                controls = np.asarray(given, dtype=np.float32)
                if controls.ndim != 3 or controls.shape[1:] != (self.total_steps, self.control):
                    raise ValueError(f"controls must have shape [candidates, {self.total_steps}, {self.control}]")
                state["action_present"] = np.ones(controls.shape[0], dtype=np.int64)
        for name in ("action_guidance", "context_guidance"):
            state[name] = float(observation.get(name, 1.0))
            if not 0.0 <= state[name] <= 20.0:
                raise ValueError(f"{name} must be between 0 and 20")
        state["samples_per_candidate"] = int(observation.get("samples_per_candidate", 1))
        if controls.shape[0] * state["samples_per_candidate"] > config.max_rows_per_request:
            raise ValueError(f"candidates x samples_per_candidate must be at most {config.max_rows_per_request}")
        rows = controls.shape[0] * state["samples_per_candidate"]
        state["return_rows"] = [int(row) for row in observation.get("return_rows", range(rows))]
        if not state["return_rows"] or not all(0 <= row < rows for row in state["return_rows"]):
            raise ValueError(f"return_rows must be a non-empty list of row indices below {rows}")
        state["has_controls"] = "controls" in observation
        state["controls"] = controls
        record_timing(batch, "validate_ms", start)
        return batch


class PrepareObservedContext(PipelineStage):
    def __init__(self, objective: WorldModelObjective, encoder: JointEncoder) -> None:
        super().__init__()
        self.objective = objective
        self.encoder = encoder

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        start = time.perf_counter()
        state = speakup_state(batch)
        device = self.objective.normalizer.video.mean.device
        if "frames" in state:
            frames = rearrange(
                torch.from_numpy(state.pop("frames")).to(device),
                "time height width channel -> 1 time channel height width",
            )
            waveform = rearrange(
                torch.from_numpy(state.pop("waveform")).to(device), "channel sample -> 1 channel sample"
            )
            video, audio = self.encoder.encode(frames.float() / 255, waveform)
        else:
            video = rearrange(
                torch.from_numpy(state.pop("context_video")).to(device), "time patch feature -> 1 time patch feature"
            )
            audio = rearrange(torch.from_numpy(state.pop("context_audio")).to(device), "time feature -> 1 time feature")
        state["context_video"] = self.objective.normalizer.normalize_video(video.float())
        state["context_audio"] = self.objective.normalizer.normalize_audio(audio.float())
        record_timing(batch, "context_ms", start)
        return batch


class EncodeCandidateControls(PipelineStage):
    def __init__(self, objective: WorldModelObjective) -> None:
        super().__init__()
        self.objective = objective

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        state = speakup_state(batch)
        device = state["context_video"].device
        conditioned = self.objective.encode_actions(
            torch.from_numpy(state.pop("controls")).to(device), torch.from_numpy(state.pop("action_present")).to(device)
        )
        state["actions"] = repeat(
            conditioned,
            "candidate time feature -> (candidate sample) time feature",
            sample=state["samples_per_candidate"],
        )
        return batch


class PrepareJointAVNoise(PipelineStage):
    def __init__(self, objective: WorldModelObjective) -> None:
        super().__init__()
        self.objective = objective

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        state = speakup_state(batch)
        context_video: Tensor = state["context_video"]
        context_audio: Tensor = state["context_audio"]
        rows = state["actions"].shape[0]
        target_steps = state["actions"].shape[1] - self.objective.context_steps
        generator = torch.Generator(device=context_video.device).manual_seed(request_seed(batch))
        state["noise_video"] = torch.randn(
            (rows, target_steps, *context_video.shape[2:]), generator=generator, device=context_video.device
        )
        state["noise_audio"] = torch.randn(
            (rows, target_steps, context_audio.shape[-1]), generator=generator, device=context_video.device
        )
        return batch


class PrepareFlowTimesteps(PipelineStage):
    def __init__(self, objective: WorldModelObjective) -> None:
        super().__init__()
        self.objective = objective

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        state = speakup_state(batch)
        state["grid"] = self.objective.transport.time_grid(batch.num_inference_steps, state["noise_video"])
        return batch


def bucket_size(rows: int, buckets: tuple[int, ...]) -> int:
    return min(bucket for bucket in buckets if bucket >= rows)


@torch.no_grad()
def denoise_rows(
    objective: WorldModelObjective,
    context_video: Tensor,
    context_audio: Tensor,
    noise_video: Tensor,
    noise_audio: Tensor,
    actions: Tensor,
    grid: Tensor,
    bucket: int,
    bfloat16: bool,
    action_guidance: float = 1.0,
    context_guidance: float = 1.0,
) -> tuple[Tensor, Tensor]:
    count = noise_video.shape[0]
    padding = bucket - count
    noise_video = torch.cat((noise_video, noise_video.new_zeros(padding, *noise_video.shape[1:])))
    noise_audio = torch.cat((noise_audio, noise_audio.new_zeros(padding, *noise_audio.shape[1:])))
    actions = torch.cat((actions, actions[:1].expand(padding, -1, -1)))
    context_video = repeat(context_video, "1 time patch feature -> rows time patch feature", rows=bucket)
    context_audio = repeat(context_audio, "1 time feature -> rows time feature", rows=bucket)
    context_steps = objective.context_steps
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bfloat16):
        predictor = objective.predictor
        if action_guidance != 1.0 or context_guidance != 1.0:
            predictor = GuidedPredictor(objective.predictor, action_guidance, context_guidance)
        sampled_video, sampled_audio = solve_flow_map(
            predictor,
            noise_video,
            noise_audio,
            context_video,
            context_audio,
            actions[:, context_steps:],
            grid,
            True,
            actions[:, :context_steps],
        )
    video = objective.transport.video_frames(sampled_video.float(), context_video)
    video = objective.normalizer.denormalize_video(video[:count])
    return video, objective.normalizer.denormalize_audio(sampled_audio[:count].float())


class SpeakUpDenoisingStage(PipelineStage):
    def __init__(self, objective: WorldModelObjective) -> None:
        super().__init__()
        self.objective = objective

    @property
    def role_affinity(self) -> RoleType:
        return RoleType.DENOISER

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        start = time.perf_counter()
        state = speakup_state(batch)
        config: SpeakUpPipelineConfig = server_args.pipeline_config
        noise_video: Tensor = state.pop("noise_video")
        noise_audio: Tensor = state.pop("noise_audio")
        actions: Tensor = state.pop("actions")
        largest = max(config.row_buckets)
        videos, audios = [], []
        for offset in range(0, noise_video.shape[0], largest):
            rows = slice(offset, offset + largest)
            video, audio = denoise_rows(
                self.objective,
                state["context_video"],
                state["context_audio"],
                noise_video[rows],
                noise_audio[rows],
                actions[rows],
                state["grid"],
                bucket_size(noise_video[rows].shape[0], config.row_buckets),
                config.bfloat16,
                state["action_guidance"] if state["has_controls"] else 1.0,
                state["context_guidance"],
            )
            videos.append(video)
            audios.append(audio)
        state["video"] = torch.cat(videos)
        state["audio"] = torch.cat(audios)
        torch.cuda.synchronize()
        record_timing(batch, "denoise_ms", start)
        return batch


class ReturnLatents(PipelineStage):
    @property
    def role_affinity(self) -> RoleType:
        return RoleType.DENOISER

    def forward(self, batch: Req, server_args: ServerArgs) -> OutputBatch:
        state = speakup_state(batch)
        rows = torch.as_tensor(state["return_rows"], device=state["video"].device)
        selected = {"video": state["video"][rows], "audio": state["audio"][rows]}
        match state["output_format"]:
            case "numpy":
                latents = {
                    name: tensor.to(torch.bfloat16).view(torch.int16).cpu().numpy() for name, tensor in selected.items()
                }
                dtype = "bfloat16"
            case "list":
                latents = {name: tensor.float().cpu().tolist() for name, tensor in selected.items()}
                dtype = "float32"
        payload = {
            "request_id": batch.request_id,
            "space": "encoder",
            "num_inference_steps": batch.num_inference_steps,
            "rows": state["return_rows"],
            "samples_per_candidate": state["samples_per_candidate"],
            "dtype": dtype,
            **latents,
        }
        if state["return_timing"]:
            payload["timings"] = state.get("timings", {})
        return OutputBatch(output=[payload], metrics=batch.metrics)
