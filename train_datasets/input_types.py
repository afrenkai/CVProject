from pathlib import Path
from pydantic import PositiveInt
from configs.config_types import FrozenModel

type InputEvent = dict[str, str | int | float | bool]


class InputStagingConfig(FrozenModel):
    directory: Path
    maximum_source_bytes: PositiveInt = 17179869184
    maximum_cache_bytes: PositiveInt = 68719476736
    minimum_free_bytes: PositiveInt = 8589934592
