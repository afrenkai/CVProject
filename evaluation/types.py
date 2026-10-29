from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, Self, TypedDict
from pydantic import Field, NonNegativeFloat, NonNegativeInt, PositiveInt, model_validator
from torch import Tensor
from configs.types import FiniteModel, FrozenModel
from training.pretraining.configuration import AllocationRuntime
from transport.types import ForecastConfig

type ForecastReview = Callable[[Tensor, Tensor], None]


class ForecastStatistics(FiniteModel):
    n_futures: PositiveInt
    mean_sample_mse: float = Field(ge=0)
    sample_mean_mse: float = Field(ge=0)
    population_sample_variance: float = Field(ge=0)
    persistence_mse: float = Field(ge=0)


class ForecastSummary(FiniteModel):
    completed: Literal[True] = True
    video: ForecastStatistics
    audio: ForecastStatistics
    n_futures: PositiveInt
    integration_steps: Literal[16] = 16
    grid_shift: Literal[4.0] = 4.0
    seeds: list[NonNegativeInt]
    recorded_action_conditioning: bool
    sampling_seconds: float = Field(ge=0)
    actions: Literal["Ground-truth historical and supplied future actions are inputs; explicit null when missing"] = (
        "Ground-truth historical and supplied future actions are inputs; explicit null when missing"
    )
    metric_space: Literal["Normalized full-feature future video and audio latents"] = (
        "Normalized full-feature future video and audio latents"
    )
    timing_scope: str = "One window, sequential forecast microbatch1; not full-corpus throughput"
    generated_waveform: Literal[False] = False

    @model_validator(mode="after")
    def complete_draws(self) -> Self:
        if (
            len(self.seeds) != self.n_futures
            or len(set(self.seeds)) != self.n_futures
            or self.video.n_futures != self.n_futures
            or self.audio.n_futures != self.n_futures
        ):
            raise ValueError("Both modalities must contain exactly n_futures distinct seeded draws")
        return self


class PrefixCheckConfig(FrozenModel):
    absolute_tolerance: NonNegativeFloat = 0.000001
    relative_tolerance: NonNegativeFloat = 0.000001


class PrefixIntervention(TypedDict):
    intervention: Literal["future_video", "future_audio", "both_futures"]
    video_maximum_absolute_difference: float
    audio_maximum_absolute_difference: float
    passed: bool


class ValidationRequest(ForecastConfig):
    global_step: PositiveInt
    checkpoint: Path
    config: Path
    validation_scope: Literal["entire_validation_split"] = "entire_validation_split"
    expected_validation_windows: PositiveInt
    test_cs_r6_used: Literal[False] = False
    training_world_size: PositiveInt = 2
    validation_world_size: PositiveInt = 2
    shard_windows: PositiveInt = 256


class ValidationConfig(FrozenModel):
    request: Path
    operation: Literal["validate", "collect"] = "validate"
    shard: NonNegativeInt = 0
    workers: NonNegativeInt = 0
    runtime: AllocationRuntime = Field(default_factory=AllocationRuntime)


class FlowMetrics(FiniteModel):
    loss: float
    video_flow_loss: float
    audio_flow_loss: float
    video_temporal_loss: float
    audio_temporal_loss: float
    unweighted_flow_loss: float
    weighted_flow_loss: float
    weighted_temporal_loss: float


class WindowIdentity(FiniteModel):
    index: NonNegativeInt
    example_seed: NonNegativeInt
    start_seconds: float
    source_id: str
    recording_id: str
    dataset: str
    recorded_action_conditioning: bool


class EvaluatedWindow(WindowIdentity):
    status: Literal["evaluated"] = "evaluated"
    flow: FlowMetrics
    forecast: ForecastSummary
    total_window_seconds: float = Field(ge=0)
    visual_example: bool


class InputRejection(WindowIdentity):
    status: Literal["native_input_rejected"] = "native_input_rejected"
    observed_global_step: PositiveInt
    error: str = Field(min_length=1)
    traceback: str = Field(min_length=1)
    raw_read_seconds: float = Field(ge=0)
    scope: Literal["Native input read only; encoder/model/numerical failures must raise"] = (
        "Native input read only; encoder/model/numerical failures must raise"
    )


@dataclass(frozen=True)
class ValidationInput:
    identity: WindowIdentity
    raw: dict[str, Tensor] | InputRejection
    started: float


type ValidationRecord = Annotated[EvaluatedWindow | InputRejection, Field(discriminator="status")]


class ValidationPart(ForecastConfig):
    global_step: PositiveInt
    checkpoint: Path
    shard: NonNegativeInt
    rank: NonNegativeInt
    world_size: PositiveInt
    expected_indices: list[NonNegativeInt]
    records: list[ValidationRecord]
    completed: bool

    @model_validator(mode="after")
    def exact_prefix(self) -> Self:
        if self.rank >= self.world_size or self.expected_indices != sorted(set(self.expected_indices)):
            raise ValueError("Invalid validation rank or candidate assignment")
        if [record.index for record in self.records] != self.expected_indices[: len(self.records)]:
            raise ValueError("Validation records must form the exact assigned prefix")
        if self.completed != (len(self.records) == len(self.expected_indices)):
            raise ValueError("Validation completion must reflect exact assigned coverage")
        return self


class ValidationEligibility(FrozenModel):
    config: Path
    windows: Path
    first_global_step: PositiveInt
    candidate_windows: PositiveInt
    eligible_indices: list[NonNegativeInt]
    rejections: list[InputRejection]
    test_cs_r6_used: Literal[False] = False


class GroupMetrics(FiniteModel):
    candidate_windows: NonNegativeInt
    evaluated_windows: NonNegativeInt
    rejected_input_windows: NonNegativeInt
    mean: dict[str, float]

    @model_validator(mode="after")
    def account_inputs(self) -> Self:
        if self.evaluated_windows + self.rejected_input_windows != self.candidate_windows:
            raise ValueError("Group metrics must account for every input candidate")
        if bool(self.mean) != (self.evaluated_windows > 0):
            raise ValueError("Only evaluated windows can have metrics")
        return self


class BestValidation(FiniteModel):
    global_step: PositiveInt
    checkpoint: Path
    unweighted_flow_loss: float
    completion: Path
    eligibility_manifest: Path
    scope: str = "Selection by full eligible validation window-mean unweighted flow loss; test sealed"


class ValidationResult(ForecastConfig):
    completed: Literal[True] = True
    global_step: PositiveInt
    checkpoint: Path
    validation_scope: Literal["entire_validation_split"] = "entire_validation_split"
    expected_validation_windows: PositiveInt
    candidate_windows_accounted: PositiveInt
    evaluated_windows: PositiveInt
    eligible_validation_windows: PositiveInt
    rejected_input_windows: NonNegativeInt
    unique_candidate_coverage_exact: Literal[True] = True
    unique_coverage_exact: Literal[True] = True
    eligibility_manifest: Path
    overall: GroupMetrics
    per_corpus: dict[str, GroupMetrics]
    examples: dict[str, NonNegativeInt]
    test_cs_r6_used: Literal[False] = False
    selected_as_best: bool

    @model_validator(mode="after")
    def exact_counts(self) -> Self:
        if self.candidate_windows_accounted != self.expected_validation_windows:
            raise ValueError("Validation completion requires every candidate window")
        if self.evaluated_windows + self.rejected_input_windows != self.candidate_windows_accounted:
            raise ValueError("Evaluated and rejected inputs must exactly account for the candidates")
        if self.eligible_validation_windows != self.evaluated_windows:
            raise ValueError("Every eligible window must be evaluated")
        expected = (self.candidate_windows_accounted, self.evaluated_windows, self.rejected_input_windows)
        overall = (self.overall.candidate_windows, self.overall.evaluated_windows, self.overall.rejected_input_windows)
        if overall != expected:
            raise ValueError("Overall metrics must agree with full-validation counts")
        corpora = (
            sum(group.candidate_windows for group in self.per_corpus.values()),
            sum(group.evaluated_windows for group in self.per_corpus.values()),
            sum(group.rejected_input_windows for group in self.per_corpus.values()),
        )
        if corpora != expected:
            raise ValueError("Corpus accounting must cover every candidate")
        if set(self.examples) != {name for name, group in self.per_corpus.items() if group.evaluated_windows}:
            raise ValueError("Each evaluated corpus needs one stable visual example")
        return self
