from pathlib import Path
import fire
from train_datasets.opencs2_events import audible, mine_events, mine_turns, silent


def main(
    output: str,
    candidates: int = 2000,
    seed: int = 0,
    directory: str = "artifacts/opencs2",
    context_seconds: float = 1.2,
    lead_seconds: float = 0.4,
    target_seconds: float = 0.8,
    task: str = "kills",
) -> None:
    if task == "turns":
        turns = mine_turns(Path(directory), candidates, seed, context_seconds, target_seconds)
        turns.to_parquet(output)
        print(
            f"{len(turns)} turn events from {candidates} POV rounds -> {output}; left share {turns['side'].mean():.2f}"
        )
        return
    events = mine_events(Path(directory), candidates, seed, context_seconds=context_seconds, lead_seconds=lead_seconds)
    events.to_parquet(output)
    print(f"{len(events)} heard-then-seen events from {candidates} kills -> {output}")
    if len(events):
        print(
            f"left share {events['side'].mean():.2f}  audible {int(audible(events).sum())}  silent {int(silent(events).sum())}"
        )
        print(events.head(3).to_string())


if __name__ == "__main__":
    fire.Fire(main)
