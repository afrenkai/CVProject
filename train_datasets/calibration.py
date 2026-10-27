from collections import defaultdict
import numpy as np
from train_datasets.corpus_types import CorpusWindow
from train_datasets.manifests import window_corpus


def calibration_windows(windows: list[CorpusWindow], count: int, seed: int) -> list[CorpusWindow]:
    assert windows and count > 0 and all(window.source.split == "train" for window in windows)
    unique = {(window.source.source_id, window.start_seconds): window for window in windows}
    candidates = sorted(unique.values(), key=lambda window: (window.source.source_id, window.start_seconds))
    groups: dict[str, list[int]] = defaultdict(list)
    for index, window in enumerate(candidates):
        groups[window_corpus(window)].append(index)
    count = min(count, len(candidates))
    if count < len(groups):
        raise ValueError("Calibration must include at least one training window per corpus")
    generator = np.random.default_rng(seed)
    selected = [int(generator.choice(indices)) for indices in groups.values()]
    chosen = set(selected)
    remaining = [index for index in range(len(candidates)) if index not in chosen]
    selected.extend(int(index) for index in generator.choice(remaining, count - len(selected), replace=False))
    return [candidates[index] for index in selected]
