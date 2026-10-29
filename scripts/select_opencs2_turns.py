from pathlib import Path
import fire
import numpy as np
import pandas as pd
from train_datasets.opencs2_events import local_path, mine_round_turns


def main(
    output: str,
    train_rounds: int = 300,
    eval_rounds: int = 75,
    eval_match_fraction: float = 0.2,
    seed: int = 0,
    directory: str = "artifacts/opencs2",
    context_seconds: float = 1.2,
    target_seconds: float = 0.8,
    repository: str = "blanchon/opencs2_dataset",
) -> None:
    rounds = pd.read_parquet(
        Path(directory) / "index" / "pov_rounds.parquet",
        columns=["match_id", "map_name", "round", "player_slot", "video", "ticks_parquet_path"],
    )
    generator = np.random.default_rng(seed)
    matches = generator.permutation(rounds["match_id"].unique())
    eval_matches = set(matches[: round(len(matches) * eval_match_fraction)].tolist())
    rounds["split"] = np.where(rounds["match_id"].isin(eval_matches), "eval", "train")
    chosen = pd.concat(
        [
            rounds[rounds["split"] == "train"].sample(n=train_rounds, random_state=seed),
            rounds[rounds["split"] == "eval"].sample(n=eval_rounds, random_state=seed),
        ]
    )
    events = []
    for row in chosen.itertuples(index=False):
        ticks = pd.read_parquet(
            local_path(repository, row.ticks_parquet_path, Path(directory)), columns=["tick", "t", "yaw", "is_alive"]
        )
        for turn in mine_round_turns(ticks, context_seconds, target_seconds, ticks_per_second=64):
            events.append(
                {
                    "match_id": row.match_id,
                    "round": row.round,
                    "player_slot": row.player_slot,
                    "split": row.split,
                    "video": row.video["path"],
                }
                | turn
            )
    frame = pd.DataFrame(events)
    frame.to_parquet(output)
    print(frame.groupby("split").agg(events=("side", "size"), left_share=("side", "mean"), videos=("video", "nunique")))


if __name__ == "__main__":
    fire.Fire(main)
