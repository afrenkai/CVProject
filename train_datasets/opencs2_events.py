from pathlib import Path
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download


def local_path(repository: str, repository_path: str, directory: Path) -> Path:
    relative = repository_path.removeprefix(f"hf://datasets/{repository}@main/")
    if (directory / relative).exists():
        return directory / relative
    return Path(hf_hub_download(repository, relative, repo_type="dataset", local_dir=directory))


def relative_bearing(viewer: pd.DataFrame, target: pd.DataFrame) -> np.ndarray:
    bearing = np.degrees(
        np.arctan2(target["y"].to_numpy() - viewer["y"].to_numpy(), target["x"].to_numpy() - viewer["x"].to_numpy())
    )
    return (bearing - viewer["yaw"].to_numpy() + 180.0) % 360.0 - 180.0


def mine_kill(
    kill: pd.Series,
    attacker_ticks: pd.DataFrame,
    victim_ticks: pd.DataFrame,
    context_seconds: float,
    lead_seconds: float,
    ticks_per_second: int,
    on_screen_degrees: float = 50.0,
    off_screen_degrees: float = 55.0,
    maximum_entry_to_kill_seconds: float = 2.0,
) -> dict | None:
    joined = attacker_ticks.merge(victim_ticks, on="tick", suffixes=("", "_victim"))
    history = joined[(joined["tick"] <= kill["tick"]) & (joined["tick"] > kill["tick"] - 4 * ticks_per_second)]
    if history.empty:
        return None
    viewer = history[["x", "y", "yaw"]]
    target = history[["x_victim", "y_victim"]].rename(columns={"x_victim": "x", "y_victim": "y"})
    all_bearings = relative_bearing(viewer, target)
    outside = np.flatnonzero(np.abs(all_bearings) >= on_screen_degrees)
    if outside.size == 0 or outside[-1] == len(all_bearings) - 1:
        return None
    entry_tick = int(history["tick"].iloc[outside[-1] + 1])
    if kill["tick"] - entry_tick > maximum_entry_to_kill_seconds * ticks_per_second:
        return None
    start_tick = entry_tick - round((context_seconds + lead_seconds) * ticks_per_second)
    context_end_tick = start_tick + round(context_seconds * ticks_per_second)
    window = history[(history["tick"] >= start_tick) & (history["tick"] <= entry_tick)]
    if window.empty or window["tick"].iloc[0] != start_tick or not window["is_alive"].all():
        return None
    bearings = all_bearings[(history["tick"] >= start_tick).to_numpy() & (history["tick"] <= entry_tick).to_numpy()]
    context = (window["tick"] <= context_end_tick).to_numpy()
    if np.abs(bearings[context]).min() <= off_screen_degrees:
        return None
    distance = np.hypot(window["x"] - window["x_victim"], window["y"] - window["y_victim"]).to_numpy()
    victim_speed = np.hypot(window["velocity_x_victim"], window["velocity_y_victim"]).to_numpy()
    return {
        "start_seconds": float(window["t"].iloc[0]),
        "entry_to_kill_seconds": float((kill["tick"] - entry_tick) / ticks_per_second),
        "side": int(np.median(bearings[context]) > 0.0),
        "context_bearing": float(np.median(bearings[context])),
        "context_distance": float(distance[context].mean()),
        "context_victim_speed": float(victim_speed[context].mean()),
        "kill_distance": float(kill["distance"]),
        "weapon_class": kill["weapon_class"],
    }


def mine_events(
    directory: Path,
    candidates: int,
    seed: int,
    context_seconds: float,
    lead_seconds: float,
    repository: str = "blanchon/opencs2_dataset",
    ticks_per_second: int = 64,
) -> pd.DataFrame:
    kills = pd.read_parquet(directory / "events" / "kills.parquet")
    kills = kills[~kills["wallbang"] & ~kills["through_smoke"] & ~kills["attacker_blind"]]
    rounds = pd.read_parquet(
        directory / "index" / "pov_rounds.parquet",
        columns=["match_id", "map_name", "round", "player_slot", "video", "ticks_parquet_path", "capture_start_tick"],
    )
    kills = kills.sample(n=min(candidates, len(kills)), random_state=seed)
    attackers = kills.merge(
        rounds,
        left_on=["match_id", "map_name", "round", "attacker_player_slot"],
        right_on=["match_id", "map_name", "round", "player_slot"],
    )
    events = []
    for row in attackers.itertuples(index=False):
        victim = rounds[
            (rounds["match_id"] == row.match_id)
            & (rounds["round"] == row.round)
            & (rounds["player_slot"] == row.victim_player_slot)
        ]
        if victim.empty:
            continue
        attacker_ticks = pd.read_parquet(local_path(repository, row.ticks_parquet_path, directory))
        victim_ticks = pd.read_parquet(
            local_path(repository, victim["ticks_parquet_path"].iloc[0], directory),
            columns=["tick", "x", "y", "velocity_x", "velocity_y"],
        )
        event = mine_kill(
            pd.Series(row._asdict()), attacker_ticks, victim_ticks, context_seconds, lead_seconds, ticks_per_second
        )
        if event is not None:
            events.append(
                {
                    "kill_id": row.kill_id,
                    "match_id": row.match_id,
                    "map_name": row.map_name,
                    "round": row.round,
                    "video": row.video,
                }
                | event
            )
    return pd.DataFrame(events)


def audible(events: pd.DataFrame, minimum_speed: float = 200.0, maximum_distance: float = 1500.0) -> pd.Series:
    return (events["context_victim_speed"] >= minimum_speed) & (events["context_distance"] <= maximum_distance)


def silent(events: pd.DataFrame, maximum_speed: float = 140.0) -> pd.Series:
    return events["context_victim_speed"] <= maximum_speed


def mine_round_turns(
    ticks: pd.DataFrame,
    context_seconds: float,
    target_seconds: float,
    ticks_per_second: int,
    still_degrees: float = 20.0,
    turn_degrees: float = 60.0,
) -> list[dict]:
    yaw = np.degrees(np.unwrap(np.radians(ticks["yaw"].to_numpy())))
    alive = ticks["is_alive"].to_numpy()
    context_ticks = round(context_seconds * ticks_per_second)
    window_ticks = context_ticks + round(target_seconds * ticks_per_second)
    turns = []
    start = 0
    while start + window_ticks <= len(ticks):
        context = yaw[start : start + context_ticks]
        turn = yaw[start + window_ticks - 1] - context[-1]
        if alive[start : start + window_ticks].all() and np.ptp(context) < still_degrees and abs(turn) > turn_degrees:
            turns.append(
                {"start_seconds": float(ticks["t"].iloc[start]), "side": int(turn > 0.0), "turn_degrees": float(turn)}
            )
            start += window_ticks
        else:
            start += ticks_per_second // 4
    return turns


def mine_turns(
    directory: Path,
    rounds_limit: int,
    seed: int,
    context_seconds: float,
    target_seconds: float,
    repository: str = "blanchon/opencs2_dataset",
    ticks_per_second: int = 64,
) -> pd.DataFrame:
    rounds = pd.read_parquet(
        directory / "index" / "pov_rounds.parquet",
        columns=["match_id", "map_name", "round", "player_slot", "video", "ticks_parquet_path"],
    ).sample(frac=1.0, random_state=seed)
    events = []
    for row in rounds.head(rounds_limit).itertuples(index=False):
        ticks = pd.read_parquet(
            local_path(repository, row.ticks_parquet_path, directory), columns=["tick", "t", "yaw", "is_alive"]
        )
        events.extend(
            {
                "match_id": row.match_id,
                "map_name": row.map_name,
                "round": row.round,
                "player_slot": row.player_slot,
                "video": row.video,
            }
            | turn
            for turn in mine_round_turns(ticks, context_seconds, target_seconds, ticks_per_second)
        )
    return pd.DataFrame(events)
