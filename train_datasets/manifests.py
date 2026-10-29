import json
from collections.abc import Sequence
from pathlib import Path
from train_datasets.corpus_types import CorpusWindow
from train_datasets.media_types import MediaRecord


def window_corpus(window: CorpusWindow) -> str:
    return "d2e" if window.source.action_present else window.source.dataset


def load_media_manifest(path: Path, split: str, excluded: Sequence[str] = ()) -> list[MediaRecord]:
    records = (MediaRecord.model_validate(record) for record in json.loads(path.read_text()))
    excluded_domains = tuple(
        "".join(character for character in name.casefold() if character.isalnum()) for name in excluded
    )
    selected: list[MediaRecord] = []
    for record in records:
        domain = "".join(character for character in record.dataset.casefold() if character.isalnum())
        if record.split == split and not domain.startswith(excluded_domains):
            selected.append(record)
    if not selected:
        raise ValueError(f"No {split} clips in {path}")
    return selected
