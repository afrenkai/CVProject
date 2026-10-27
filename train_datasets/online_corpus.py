import json
import os
import random
from collections import defaultdict
from pathlib import Path
from torch import Tensor
from train_datasets.corpus_dataset import CorpusWindows, read_corpus_window
from train_datasets.corpus_types import CorpusConfig, CorpusWindow
from train_datasets.input_types import InputEvent
from train_datasets.manifests import window_corpus


class OnlineCorpusWindows(CorpusWindows):
    def __init__(
        self,
        windows: list[CorpusWindow],
        config: CorpusConfig,
        journal_directory: Path,
        seed: int,
        read_attempts: int = 64,
    ) -> None:
        assert windows and all(window.source.split == "train" for window in windows) and read_attempts > 0
        super().__init__(windows, config)
        self.seed = seed
        self.read_attempts = read_attempts
        self.groups: dict[str, list[int]] = defaultdict(list)
        for index, window in enumerate(windows):
            self.groups[window_corpus(window)].append(index)
        self.journal_directory = journal_directory
        journal_directory.mkdir(parents=True, exist_ok=True)

    def record(self, event: InputEvent) -> None:
        job = os.environ.get("SLURM_JOB_ID", "local")
        rank = os.environ.get("RANK", "0")
        path = self.journal_directory / f"job_{job}_rank_{rank}_worker_{os.getpid()}.jsonl"
        with path.open("a") as stream:
            stream.write(json.dumps(event) + "\n")

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        requested = self.windows[index]
        corpus = window_corpus(requested)
        generator = random.Random(requested.example_seed + self.seed)
        candidate_index = index
        for attempt in range(self.read_attempts):
            window = self.windows[candidate_index]
            identity: InputEvent = {
                "requested_seed": requested.example_seed,
                "source": window.source.source_id,
                "start_seconds": window.start_seconds,
            }
            try:
                batch = read_corpus_window(window, self.config)
            except (AssertionError, RuntimeError, ValueError, OSError, IndexError) as error:
                self.record(
                    identity
                    | {
                        "event": "input_rejected",
                        "candidate_seed": window.example_seed,
                        "reason": f"{type(error).__name__}: {error}",
                    }
                )
                candidate_index = generator.choice(self.groups[corpus])
                continue
            self.record(
                identity
                | {
                    "event": "input_loaded",
                    "actual_seed": window.example_seed,
                    "replacement": attempt > 0,
                    "scope": "Loader event; prefetch may precede optimizer consumption",
                }
            )
            return batch
        raise RuntimeError(f"No readable {corpus} input after {self.read_attempts} native guarded attempts")
