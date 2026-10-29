import fcntl
import shutil
from pathlib import Path
from train_datasets.corpus_types import SourceRecord
from train_datasets.input_types import InputStagingConfig


def staged_media_path(source: SourceRecord, config: InputStagingConfig | None) -> Path:
    if config is None or not source.action_present:
        return source.path
    assert source.split in {"train", "validation"}
    relative = Path(source.source_id)
    assert not relative.is_absolute() and ".." not in relative.parts
    size = source.path.stat().st_size
    if size > config.maximum_source_bytes:
        return source.path
    config.directory.mkdir(parents=True, exist_ok=True)
    destination = config.directory / relative
    with (config.directory / "staging.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if destination.exists():
            assert destination.stat().st_size == size
            return destination
        pending = destination.with_name(destination.name + ".partial")
        used = sum(entry.stat().st_size for entry in config.directory.rglob("*") if entry.is_file())
        if pending.exists() or used + size > config.maximum_cache_bytes:
            return source.path
        if shutil.disk_usage(config.directory).free - size < config.minimum_free_bytes:
            return source.path
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            with source.path.open("rb") as original, pending.open("xb") as staged:
                shutil.copyfileobj(original, staged, length=8388608)
        except OSError:
            return source.path
        assert pending.stat().st_size == size
        pending.replace(destination)
        return destination
