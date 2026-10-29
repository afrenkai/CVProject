import math
import os
import time
from collections import defaultdict
from functools import partial
from pathlib import Path
from traceback import format_exc
import hydra
import torch
import yaml
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import load_model
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset
from architecture.pretraining.joint_encoder import JointEncoder
from evaluation.evaluation_types import (
    BestValidation,
    EvaluatedWindow,
    FlowMetrics,
    GroupMetrics,
    InputRejection,
    ValidationConfig,
    ValidationEligibility,
    ValidationInput,
    ValidationPart,
    ValidationRecord,
    ValidationRequest,
    ValidationResult,
    WindowIdentity,
)
from evaluation.forecast import forecast_window
from plotting.validation import render_sample_mean, render_validation_summary
from train_datasets.corpus_dataset import read_corpus_window
from train_datasets.corpus_types import CorpusWindow
from train_datasets.manifests import window_corpus
from train_datasets.records import read_windows, write_yaml
from training.pretraining.allocation import allocation_deadline
from training.pretraining.configuration import PretrainingConfig
from training.pretraining.frozen import build_online_objective
from training.pretraining.schedule import native_checkpoint_step
from training.runtime import cuda_device


def window_identity(index: int, window: CorpusWindow) -> WindowIdentity:
    assert window.source.split == "validation"
    return WindowIdentity(
        index=index,
        example_seed=window.example_seed,
        start_seconds=window.start_seconds,
        source_id=window.source.source_id,
        recording_id=window.source.recording_id,
        dataset=window_corpus(window),
        recorded_action_conditioning=window.source.action_present,
    )


def assert_record_identity(record: ValidationRecord, window: CorpusWindow) -> None:
    identity = WindowIdentity.model_validate(record.model_dump(include=set(WindowIdentity.model_fields)))
    assert identity == window_identity(record.index, window)


def assert_record_protocol(record: ValidationRecord, request: ValidationRequest) -> None:
    if isinstance(record, EvaluatedWindow):
        assert record.forecast.n_futures == request.n_futures
        assert record.forecast.integration_steps == request.integration_steps
        assert record.forecast.grid_shift == request.grid_shift
        assert record.forecast.recorded_action_conditioning == record.recorded_action_conditioning


def assert_eligibility_coverage(eligibility: ValidationEligibility, windows: list[CorpusWindow]) -> None:
    eligible = eligibility.eligible_indices
    rejected = [record.index for record in eligibility.rejections]
    assert eligibility.candidate_windows == len(windows)
    assert eligible and eligible == sorted(set(eligible)) and rejected == sorted(set(rejected))
    assert set(eligible).isdisjoint(rejected) and set(eligible).union(rejected) == set(range(len(windows)))
    for record in eligibility.rejections:
        assert record.observed_global_step == eligibility.first_global_step
        assert_record_identity(record, windows[record.index])


def read_pretraining_config(path: Path) -> PretrainingConfig:
    config = instantiate(OmegaConf.load(path))
    assert isinstance(config, PretrainingConfig)
    return config


def restore_part(path: Path, request: ValidationRequest, shard: int, rank: int) -> ValidationPart:
    assert 0 <= shard < math.ceil(request.expected_validation_windows / request.shard_windows)
    assert 0 <= rank < request.validation_world_size
    begin = shard * request.shard_windows
    stop = min(begin + request.shard_windows, request.expected_validation_windows)
    expected = list(range(begin + rank, stop, request.validation_world_size))
    if path.exists():
        part = ValidationPart.model_validate(yaml.load(path.read_text(), Loader=yaml.CSafeLoader))
        assert part.global_step == request.global_step and part.checkpoint == request.checkpoint
        assert part.shard == shard and part.rank == rank and part.world_size == request.validation_world_size
        assert part.expected_indices == expected
        assert part.n_futures == request.n_futures
        return part
    return ValidationPart(
        global_step=request.global_step,
        checkpoint=request.checkpoint,
        shard=shard,
        rank=rank,
        world_size=request.validation_world_size,
        n_futures=request.n_futures,
        expected_indices=expected,
        records=[],
        completed=not expected,
    )


class WindowAverages:
    def __init__(self) -> None:
        self.candidates = 0
        self.evaluated = 0
        self.sums: dict[str, float] = defaultdict(float)

    def update(self, record: ValidationRecord) -> None:
        self.candidates += 1
        if not isinstance(record, EvaluatedWindow):
            return
        values = {f"flow/{name}": value for name, value in record.flow.model_dump().items()}
        for modality, statistics in (("video", record.forecast.video), ("audio", record.forecast.audio)):
            values.update(
                {f"{modality}/{name}": value for name, value in statistics.model_dump(exclude={"n_futures"}).items()}
            )
        values["total_window_seconds"] = record.total_window_seconds
        for name, value in values.items():
            self.sums[name] += value
        self.evaluated += 1

    def summary(self) -> GroupMetrics:
        return GroupMetrics(
            candidate_windows=self.candidates,
            evaluated_windows=self.evaluated,
            rejected_input_windows=self.candidates - self.evaluated,
            mean={name: value / self.evaluated for name, value in self.sums.items()},
        )


class ValidationInputs(Dataset[ValidationInput]):
    def __init__(
        self,
        windows: list[CorpusWindow],
        config: PretrainingConfig,
        request: ValidationRequest,
        eligibility: ValidationEligibility | None,
    ) -> None:
        self.windows, self.config, self.request = windows, config, request
        self.eligibility_frozen = eligibility is not None
        self.rejections = {} if eligibility is None else {record.index: record for record in eligibility.rejections}

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> ValidationInput:
        window = self.windows[index]
        identity = window_identity(index, window)
        started = time.perf_counter()
        raw = self.rejections.get(index)
        if raw is None:
            try:
                raw = read_corpus_window(window, self.config.data)
            except (AssertionError, RuntimeError, ValueError, OSError, IndexError) as error:
                if isinstance(error, torch.OutOfMemoryError):
                    raise
                if self.eligibility_frozen:
                    raise RuntimeError(f"Previously eligible validation input failed at index {index}") from error
                raw = InputRejection(
                    **identity.model_dump(),
                    observed_global_step=self.request.global_step,
                    error=f"{type(error).__name__}: {error}",
                    traceback=format_exc(),
                    raw_read_seconds=time.perf_counter() - started,
                )
        return ValidationInput(identity, raw, started)


class ValidationEvaluator:
    def __init__(
        self,
        request: ValidationRequest,
        config: PretrainingConfig,
        directory: Path,
        device: torch.device,
        records: list[ValidationRecord],
    ) -> None:
        self.request, self.config, self.directory, self.device = request, config, directory, device
        self.rendered = {
            record.dataset for record in records if isinstance(record, EvaluatedWindow) and record.visual_example
        }
        self.encoder = JointEncoder(config.encoder, device)
        self.objective = build_online_objective(config, self.encoder.encode, device).eval().requires_grad_(False)
        load_model(self.objective, request.checkpoint / "model.safetensors", strict=True, device=str(device))

    @torch.no_grad()
    def evaluate(
        self, identity: WindowIdentity, raw: dict[str, Tensor], visual: bool, started: float
    ) -> EvaluatedWindow:
        config, request, objective = self.config, self.request, self.objective
        batch = {name: value.unsqueeze(0).to(self.device) for name, value in raw.items()}
        video, audio = self.encoder.encode(batch["frames"], batch["waveform"])
        video, audio = objective.normalizer.normalize_video(video), objective.normalizer.normalize_audio(audio)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = objective.feature_loss(
                video, audio, batch["actions"], batch["action_present"], batch["example_seed"]
            )
        flow = FlowMetrics.model_validate({name: value.float().mean().item() for name, value in loss.items()})
        assert math.isclose(
            flow.unweighted_flow_loss, flow.video_flow_loss + flow.audio_flow_loss, rel_tol=0.00001, abs_tol=0.000001
        )
        renderer = (
            partial(
                render_sample_mean,
                directory=self.directory / f"example_{identity.index:07d}",
                video=video,
                audio=audio,
                context=objective.context_steps,
                config=config.encoder,
                seed=config.trainer.seed,
                n_futures=request.n_futures,
            )
            if visual
            else None
        )
        forecast = forecast_window(
            objective,
            video,
            audio,
            batch["actions"],
            batch["action_present"],
            identity.example_seed,
            request,
            config.trainer.seed,
            renderer,
        )
        return EvaluatedWindow(
            **identity.model_dump(),
            flow=flow,
            forecast=forecast,
            total_window_seconds=time.perf_counter() - started,
            visual_example=visual,
        )

    def record(self, sample: ValidationInput) -> ValidationRecord:
        identity, raw = sample.identity, sample.raw
        if isinstance(raw, InputRejection):
            return raw
        record = self.evaluate(identity, raw, identity.dataset not in self.rendered, sample.started)
        if record.visual_example:
            self.rendered.add(record.dataset)
        return record


@torch.no_grad()
def run_validation(settings: ValidationConfig) -> None:
    request = ValidationRequest.model_validate(yaml.load(settings.request.read_text(), Loader=yaml.CSafeLoader))
    assert native_checkpoint_step(request.checkpoint, request.training_world_size) == request.global_step
    config = read_pretraining_config(request.config)
    rank, world_size = int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    assert world_size == request.validation_world_size == torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    device = cuda_device(config.trainer.seed)
    assert device.index == local_rank and torch.cuda.is_bf16_supported()
    windows = read_windows(config.windows.validation, "validation")
    assert len(windows) == request.expected_validation_windows
    directory = settings.request.parent / f"step_{request.global_step}" / f"n_futures_{request.n_futures}"
    destination = directory / f"shard_{settings.shard:05d}" / f"rank_{rank}.yaml"
    part = restore_part(destination, request, settings.shard, rank)
    for record in part.records:
        assert_record_identity(record, windows[record.index])
        assert_record_protocol(record, request)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if part.completed:
        write_yaml(destination, part.model_dump(mode="json"))
        print("already_completed", destination, flush=True)
        return
    eligibility_path = settings.request.parent / "validation_eligibility.yaml"
    eligibility = None
    if eligibility_path.exists():
        eligibility = ValidationEligibility.model_validate(
            yaml.load(eligibility_path.read_text(), Loader=yaml.CSafeLoader)
        )
        assert eligibility.windows == config.windows.validation and eligibility.first_global_step <= request.global_step
        assert_eligibility_coverage(eligibility, windows)
    deadline = allocation_deadline(settings.runtime)
    assert deadline is None or time.time() < deadline
    assert settings.runtime.stop_file is None or not settings.runtime.stop_file.exists()
    evaluator = ValidationEvaluator(request, config, destination.parent, device, part.records)
    records = list(part.records)
    inputs = ValidationInputs(windows, config, request, eligibility)
    remaining = Subset(inputs, part.expected_indices[len(records) :])
    for sample in DataLoader(remaining, batch_size=None, num_workers=settings.workers):
        expired = deadline is not None and time.time() >= deadline
        requested = settings.runtime.stop_file is not None and settings.runtime.stop_file.exists()
        if expired or requested:
            break
        index = sample.identity.index
        record = evaluator.record(sample)
        assert_record_identity(record, windows[index])
        assert_record_protocol(record, request)
        records.append(record)
        part = part.model_copy(update={"records": records, "completed": len(records) == len(part.expected_indices)})
        write_yaml(destination, part.model_dump(mode="json"))
        print("validation_candidate_accounted", request.global_step, index, len(windows), record.status, flush=True)
    print(
        "validation_part_progress",
        request.global_step,
        settings.shard,
        rank,
        len(records),
        len(part.expected_indices),
        flush=True,
    )


def collect_records(directory: Path, request: ValidationRequest, windows: list[CorpusWindow]) -> list[ValidationRecord]:
    records: dict[int, ValidationRecord] = {}
    for shard in range(math.ceil(len(windows) / request.shard_windows)):
        for rank in range(request.validation_world_size):
            destination = directory / f"shard_{shard:05d}" / f"rank_{rank}.yaml"
            assert destination.is_file(), destination
            part = restore_part(destination, request, shard, rank)
            assert part.completed, destination
            for record in part.records:
                assert record.index not in records
                assert_record_identity(record, windows[record.index])
                assert_record_protocol(record, request)
                records[record.index] = record
    assert set(records) == set(range(len(windows)))
    return [records[index] for index in range(len(windows))]


def resolve_eligibility(
    path: Path,
    request: ValidationRequest,
    windows_path: Path,
    windows: list[CorpusWindow],
    records: list[ValidationRecord],
) -> ValidationEligibility:
    eligible = [record.index for record in records if isinstance(record, EvaluatedWindow)]
    rejections = [record for record in records if isinstance(record, InputRejection)]
    if path.exists():
        eligibility = ValidationEligibility.model_validate(yaml.load(path.read_text(), Loader=yaml.CSafeLoader))
        assert eligibility.windows == windows_path and eligibility.first_global_step <= request.global_step
        assert eligibility.eligible_indices == eligible and eligibility.rejections == rejections
    else:
        eligibility = ValidationEligibility(
            config=request.config,
            windows=windows_path,
            first_global_step=request.global_step,
            candidate_windows=len(windows),
            eligible_indices=eligible,
            rejections=rejections,
        )
    assert_eligibility_coverage(eligibility, windows)
    return eligibility


def stable_examples(directory: Path, request: ValidationRequest, records: list[ValidationRecord]) -> dict[str, int]:
    examples: dict[str, int] = {}
    for record in records:
        if not isinstance(record, EvaluatedWindow) or record.dataset in examples:
            continue
        assert record.visual_example
        example = directory / f"shard_{record.index // request.shard_windows:05d}" / f"example_{record.index:07d}"
        assert all((example / name).is_file() for name in ("video_latent_pca.png", "audio_latents.png"))
        examples[record.dataset] = record.index
    return examples


def run_collection(settings: ValidationConfig) -> ValidationResult:
    request = ValidationRequest.model_validate(yaml.load(settings.request.read_text(), Loader=yaml.CSafeLoader))
    config = read_pretraining_config(request.config)
    windows = read_windows(config.windows.validation, "validation")
    assert len(windows) == request.expected_validation_windows
    directory = settings.request.parent / f"step_{request.global_step}" / f"n_futures_{request.n_futures}"
    records = collect_records(directory, request, windows)
    eligibility_path = settings.request.parent / "validation_eligibility.yaml"
    eligibility = resolve_eligibility(eligibility_path, request, config.windows.validation, windows, records)
    overall = WindowAverages()
    corpora: dict[str, WindowAverages] = defaultdict(WindowAverages)
    sources: dict[str, WindowAverages] = defaultdict(WindowAverages)
    recordings: dict[str, WindowAverages] = defaultdict(WindowAverages)
    for record in records:
        for accumulator in (
            overall,
            corpora[record.dataset],
            sources[record.source_id],
            recordings[record.recording_id],
        ):
            accumulator.update(record)
    summary = overall.summary()
    groups = {
        level: {name: accumulator.summary() for name, accumulator in accumulators.items()}
        for level, accumulators in (("per_corpus", corpora), ("per_source", sources), ("per_recording", recordings))
    }
    current = BestValidation(
        global_step=request.global_step,
        checkpoint=request.checkpoint,
        unweighted_flow_loss=summary.mean["flow/unweighted_flow_loss"],
        completion=directory / "completed.yaml",
        eligibility_manifest=eligibility_path,
    )
    selector = settings.request.parent / "best_validation.yaml"
    previous = (
        BestValidation.model_validate(yaml.load(selector.read_text(), Loader=yaml.CSafeLoader))
        if selector.exists()
        else None
    )
    assert previous is None or previous.eligibility_manifest == eligibility_path
    selected = (
        previous is None
        or previous.global_step == request.global_step
        or current.unweighted_flow_loss < previous.unweighted_flow_loss
    )
    result = ValidationResult(
        n_futures=request.n_futures,
        global_step=request.global_step,
        checkpoint=request.checkpoint,
        expected_validation_windows=len(windows),
        candidate_windows_accounted=len(records),
        evaluated_windows=summary.evaluated_windows,
        eligible_validation_windows=len(eligibility.eligible_indices),
        rejected_input_windows=summary.rejected_input_windows,
        eligibility_manifest=eligibility_path,
        overall=summary,
        per_corpus=groups["per_corpus"],
        examples=stable_examples(directory, request, records),
        selected_as_best=selected,
    )
    accounting = {
        level: {name: metrics.model_dump() for name, metrics in summaries.items()}
        for level, summaries in groups.items()
    }
    for level in ("per_source", "per_recording"):
        write_yaml(directory / f"{level}.yaml", accounting[level])
    write_yaml(directory / "windows.yaml", [record.model_dump(mode="json") for record in records])
    write_yaml(
        directory / "input_accounting.yaml",
        {
            "candidate_windows": len(records),
            "evaluated_windows": summary.evaluated_windows,
            "rejected_input_windows": summary.rejected_input_windows,
            **accounting,
        },
    )
    render_validation_summary(directory, result)
    if not eligibility_path.exists():
        write_yaml(eligibility_path, eligibility.model_dump(mode="json"))
    if selected:
        write_yaml(selector, current.model_dump(mode="json"))
    write_yaml(directory / "completed.yaml", result.model_dump(mode="json"))
    return result


@hydra.main(config_path="../configs", config_name="validation", version_base=None)
def main(config: DictConfig) -> None:
    settings = instantiate(config)
    assert isinstance(settings, ValidationConfig)
    if settings.operation == "collect":
        print(yaml.safe_dump(run_collection(settings).model_dump(mode="json"), sort_keys=False), flush=True)
    else:
        run_validation(settings)


if __name__ == "__main__":
    main()
