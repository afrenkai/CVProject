import argparse
from pathlib import Path
import numpy as np
import pandas
import pyarrow.parquet
from data.data_types import CueConfig, RoundState


def round_state(frame: pandas.DataFrame, config: CueConfig) -> RoundState:
    ticks = np.sort(frame.tick.unique())
    slots = np.arange(10)
    grid = frame.set_index(["tick", "player_slot"]).reindex(pandas.MultiIndex.from_product([ticks, slots]))

    def plane(column: str, fill: float) -> np.ndarray:
        return grid[column].to_numpy(dtype=np.float64, na_value=fill).reshape(len(ticks), len(slots))

    active = grid["active"].to_numpy()
    firing = np.array([isinstance(names, list | np.ndarray) and "attack" in names for names in active])
    walking = np.array(
        [isinstance(names, list | np.ndarray) and ("speed" in names or "duck" in names) for names in active]
    )
    speed = np.hypot(plane("velocity_x", 0.0), plane("velocity_y", 0.0))
    return RoundState(
        ticks=ticks,
        position=np.stack((plane("x", np.nan), plane("y", np.nan), plane("z", np.nan)), axis=-1),
        yaw=plane("yaw", 0.0),
        pitch=plane("pitch", 0.0),
        team=plane("team_num", -1.0),
        alive=plane("is_alive", 0.0) > 0.5,
        firing=firing.reshape(len(ticks), len(slots)),
        running=(speed > config.running_speed) & ~walking.reshape(len(ticks), len(slots)),
    )


def relative_angles(state: RoundState, viewer: int, config: CueConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eye = state.position[:, viewer : viewer + 1] + np.array([0.0, 0.0, config.eye_height])
    offset = state.position + np.array([0.0, 0.0, config.eye_height * 0.8]) - eye
    distance = np.linalg.norm(offset, axis=-1)
    bearing = np.degrees(np.arctan2(offset[..., 1], offset[..., 0]))
    elevation = np.degrees(np.arctan2(-offset[..., 2], np.hypot(offset[..., 0], offset[..., 1])))
    yaw_error = (bearing - state.yaw[:, viewer : viewer + 1] + 180.0) % 360.0 - 180.0
    pitch_error = elevation - state.pitch[:, viewer : viewer + 1]
    return yaw_error, pitch_error, distance


def pov_windows(
    state: RoundState, viewer: int, capture_start: int, last_tick: int, kills: pandas.DataFrame, config: CueConfig
) -> list[dict[str, object]]:
    yaw_error, pitch_error, distance = relative_angles(state, viewer, config)
    enemies = (state.team != state.team[:, viewer : viewer + 1]) & (state.team >= 0) & state.alive
    in_view = enemies & (np.abs(yaw_error) < config.view_half_yaw) & (np.abs(pitch_error) < config.view_half_pitch)
    aimed = np.hypot(yaw_error, pitch_error) < config.aim_degrees
    engaged = enemies & aimed & (distance < config.engage_distance) & state.firing[:, viewer : viewer + 1]
    for kill in kills.itertuples():
        index = np.searchsorted(state.ticks, kill.tick)
        if kill.attacker_player_slot == viewer and index < len(state.ticks):
            engaged[index, kill.victim_player_slot] = True
    audible = enemies & (
        (state.firing & (distance < config.gunshot_distance)) | (state.running & (distance < config.footstep_distance))
    )
    rate = config.tick_rate
    context, target = round(config.context_seconds * rate), round(config.target_seconds * rate)
    lead = context + round(config.onset_into_target * rate)
    usable = (state.ticks >= capture_start) & (state.ticks <= last_tick) & state.alive[:, viewer]
    windows = []
    for enemy in range(10):
        onsets = np.flatnonzero(engaged[:, enemy])
        gap = round(config.quiet_gap_seconds * rate)
        for onset in onsets[np.diff(onsets, prepend=-gap - 1) > gap]:
            begin, end = onset - lead, onset - lead + context + target
            if begin < 0 or end > len(state.ticks) or not usable[begin:end].all():
                continue
            if in_view[begin : begin + context, enemy].any() or engaged[begin : begin + context].any():
                continue
            heard = audible[begin : begin + context, enemy]
            gunshot = (state.firing[begin : begin + context, enemy] & heard).any()
            windows.append(
                {
                    "start": float((state.ticks[begin] - capture_start) / rate),
                    "label": "cue" if heard.any() else "silent",
                    "cue": "gunshot" if gunshot else "footsteps" if heard.any() else "",
                    "enemy_slot": enemy,
                    "distance": float(distance[onset, enemy]),
                    "context_yaw_error": float(yaw_error[begin + context - 1, enemy]),
                }
            )
    quiet = ~audible.any(axis=1) & ~engaged.any(axis=1)
    generator = np.random.default_rng(config.seed + viewer)
    candidates = np.arange(0, len(state.ticks) - context - target, rate)
    generator.shuffle(candidates)
    picked = 0
    for begin in candidates:
        end = begin + context + target
        if picked == config.none_per_pov:
            break
        if usable[begin:end].all() and quiet[begin:end].all():
            windows.append(
                {
                    "start": float((state.ticks[begin] - capture_start) / rate),
                    "label": "none",
                    "cue": "",
                    "enemy_slot": -1,
                    "distance": float("nan"),
                    "context_yaw_error": float("nan"),
                }
            )
            picked += 1
    return windows


def label_match(
    path: Path, povs: pandas.DataFrame, kills: pandas.DataFrame, config: CueConfig
) -> list[dict[str, object]]:
    columns = ["round", "player_slot", "tick", "active", "team_num", "is_alive", "x", "y", "z", "yaw", "pitch"]
    frame = pyarrow.parquet.read_table(path, columns=[*columns, "velocity_x", "velocity_y"]).to_pandas()
    rows = []
    for round_number, round_frame in frame.groupby("round"):
        round_povs = povs[povs["round"] == round_number]
        if round_povs.empty:
            continue
        state = round_state(round_frame, config)
        round_kills = kills[kills["round"] == round_number]
        for pov in round_povs.itertuples():
            last_tick = int(pov.death_tick) if pandas.notna(pov.death_tick) else int(pov.capture_end_tick)
            for window in pov_windows(
                state, int(pov.player_slot), int(pov.capture_start_tick), last_tick, round_kills, config
            ):
                rows.append({"media_id": pov.media_id, **window})
    return rows


def main(root: Path, split: str, output: Path, config: CueConfig) -> None:
    keys = (root / "splits" / "opencs2" / f"{split}.txt").read_text().split()
    povs = pyarrow.parquet.read_table(root / "index" / "pov_rounds.parquet").to_pandas()
    povs = povs[povs.media_id.isin(keys)]
    kills = pyarrow.parquet.read_table(root / "events" / "kills.parquet").to_pandas()
    rows = []
    for (match_id, map_name), match_povs in povs.groupby(["match_id", "map_name"]):
        path = root / "ticks" / f"match_id={match_id}" / f"map_name={map_name}" / "ticks.parquet"
        match_kills = kills[(kills.match_id == match_id) & (kills.map_name == map_name)]
        rows.extend(label_match(path, match_povs, match_kills, config))
        print(f"{match_id} {map_name}: {len(rows)} windows", flush=True)
    windows = pandas.DataFrame(rows)
    print(windows.groupby(["label", "cue"]).size().to_string(), flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    windows.to_parquet(output, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--split", default="validation")
    args = parser.parse_args()
    main(args.root, args.split, args.output, CueConfig())
