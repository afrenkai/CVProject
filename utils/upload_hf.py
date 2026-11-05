import argparse
import json
from pathlib import Path
from huggingface_hub import HfApi
from transformers.trainer_utils import get_last_checkpoint
from training.objective import checkpoint_weights
from utils.hub import SpeakUp


def load_run(run: Path, use_ema: bool) -> SpeakUp:
    settings = json.loads((run / "pretrain_config.json").read_text())
    checkpoint = get_last_checkpoint(str(run))
    assert checkpoint is not None, f"No checkpoint-* directory in {run}"
    model = SpeakUp({name: settings[name] for name in ("encoder", "model", "objective")})
    model.objective.load_state_dict(checkpoint_weights(Path(checkpoint), use_ema))
    print(f"Loaded {'EMA' if use_ema else 'raw'} weights from {checkpoint}", flush=True)
    return model.eval()


def main(run: Path, repo_id: str, push: bool, use_ema: bool) -> None:
    model = load_run(run, use_ema)
    folder = run / "hub"
    model.save_pretrained(folder)
    SpeakUp.from_pretrained(folder)
    files = {path.name: path.stat().st_size for path in folder.iterdir()}
    print(f"Saved and reloaded {folder}: {files}", flush=True)
    api = HfApi()
    print(f"Authenticated as {api.whoami()['name']}", flush=True)
    if not push:
        print(f"Dry run: would push {list(files)} to private repo {repo_id}", flush=True)
        return
    url = model.push_to_hub(repo_id, private=True)
    print(f"Pushed to {url}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("repo_id")
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--raw", action="store_true")
    command = parser.parse_args()
    main(command.run, command.repo_id, command.push, not command.raw)
