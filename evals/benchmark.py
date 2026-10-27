from pathlib import Path
from torch import Tensor
from torch.utils.data import Dataset


class Benchmark(Dataset[dict[str, Tensor]]):
    def __init__(self, root: Path) -> None:
        self.root = root

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        raise NotImplementedError


class AudioSet20K(Benchmark):
    pass


class AudioSet2M(Benchmark):
    pass


class ESC50(Benchmark):
    pass


class FSD50K(Benchmark):
    pass


class Kinetics400(Benchmark):
    pass
