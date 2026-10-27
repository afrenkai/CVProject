from pathlib import Path
from pydantic import Field, NonNegativeInt, PositiveFloat, PositiveInt
from evaluation.types import PrefixCheckConfig
from training.pretraining.configuration import FrozenInputConfig


class NormalizationConfig(FrozenInputConfig):
    windows: Path
    output: Path
    sample_windows: PositiveInt = 1024
    workers: NonNegativeInt = 2
    seed: NonNegativeInt = 67
    minimum_standard_deviation: PositiveFloat = 0.000001
    prefix_checks: PrefixCheckConfig = Field(default_factory=PrefixCheckConfig)
