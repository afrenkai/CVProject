from pathlib import Path
from typing import Literal
import yaml
from train_datasets.corpus_types import CorpusWindow


def write_yaml(path: Path, value: object) -> None:
    pending = path.with_suffix(".pending")
    pending.write_text(yaml.dump(value, Dumper=yaml.CSafeDumper, sort_keys=False))
    pending.replace(path)


def read_windows(path: Path, split: Literal["train", "validation"]) -> list[CorpusWindow]:
    rows = yaml.load(path.read_text(), Loader=yaml.CSafeLoader)
    assert isinstance(rows, list) and rows
    windows = [CorpusWindow.model_validate(row) for row in rows]
    assert all(window.source.split == split for window in windows)
    return windows
