from pathlib import Path
from evals.benchmark import ESC50, FSD50K, AudioSet2M, AudioSet20K, Kinetics400

if __name__ == "__main__":
    for dataset in (
        AudioSet20K(Path("/path/to/as20k")),
        AudioSet2M(Path("/path/to/as2m")),
        ESC50(Path("/path/to/esc50")),
        FSD50K(Path("/path/to/fsd50k")),
        Kinetics400(Path("/path/to/k400")),
    ):
        print(next(iter(dataset)))
