import argparse
import ast
import re
import shutil
from pathlib import Path
from huggingface_hub import HfApi
from safetensors.torch import save_model
from transformers import AutoModel
from transformers.trainer_utils import get_last_checkpoint
from training.objective import checkpoint_weights
from training.training_types import PretrainConfig
from utils.hub import SpeakUp
from utils.hub_types import SpeakUpConfig


def module_path(module: str) -> Path:
    return Path(*module.split(".")).with_suffix(".py")


def flat_name(module: str) -> str:
    return module.replace(".", "_")


def local_imports(module: str) -> list[str]:
    return [
        node.module
        for node in ast.walk(ast.parse(module_path(module).read_text()))
        if isinstance(node, ast.ImportFrom) and node.module is not None and module_path(node.module).is_file()
    ]


def source_closure(root: str) -> set[str]:
    modules: set[str] = set()
    pending = [root]
    while pending:
        module = pending.pop()
        if module not in modules:
            modules.add(module)
            pending.extend(local_imports(module))
    return modules


def vendored_source(module: str, modules: set[str]) -> str:
    source = module_path(module).read_text()
    for imported in modules:
        source = re.sub(
            rf"^(\s*)from {re.escape(imported)} import",
            rf"\1from .{flat_name(imported)} import",
            source,
            flags=re.MULTILINE,
        )
    assert not local_imports_in(source, modules), f"{module} still imports repository modules"
    return source


def local_imports_in(source: str, modules: set[str]) -> list[str]:
    return [
        node.module
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module in modules
    ]


def model_card(repo_id: str, checkpoint: Path, use_ema: bool) -> str:
    return f"""---
library_name: transformers
tags: [world-model, audio-visual, flow-matching, mixture-of-transformers, custom_code]
---

# {repo_id.split("/")[-1]}

Action-conditioned audio-visual latent world model (SpeakUp), {"EMA" if use_ema else "raw"} weights from `{checkpoint.name}`.
Frozen encoders load from their own sources at init: DINOv3 through timm and Dasheng through `mispeech` remote code.

```python
from transformers import AutoModel

model = AutoModel.from_pretrained("{repo_id}", trust_remote_code=True)
video, audio = model.encode(frames, waveform)
prediction = model.rollout(video, audio, actions, action_present)
```

- `frames`: `[batch, frames, 3, height, width]` in `[0, 1]`, `model.encoder_config.video_rate` fps.
- `waveform`: `[batch, channels, samples]` at `model.encoder_config.loading_sample_rate` Hz.
- `actions`: recorded controls `[batch, steps, control]`; `action_present` zeros select the learned null action.
- `rollout` keeps the first `model.objective.context_steps` steps as context and samples the rest.

Requires `torch`, `torchaudio`, `timm`, `diffusers`, `einops`, `jaxtyping`, `beartype`, `pydantic`, `safetensors`.
"""


def load_run(run: Path, use_ema: bool) -> tuple[SpeakUp, Path]:
    settings = PretrainConfig.model_validate_json((run / "config.json").read_text())
    checkpoint = get_last_checkpoint(str(run))
    assert checkpoint is not None, f"No checkpoint-* directory in {run}"
    config = SpeakUpConfig(
        encoder=settings.encoder.model_dump(mode="json", exclude={"video": {"cache_directory"}}),
        model=settings.model.model_copy(update={"attention_backend": "sdpa"}).model_dump(mode="json"),
        objective=settings.objective.model_dump(
            mode="json", exclude={"slim_lsd", "action_dropout", "modality_dropout"}
        ),
        encoder_bfloat16=settings.trainer.precision == "bf16",
        auto_map={
            "AutoConfig": f"{flat_name(SpeakUpConfig.__module__)}.{SpeakUpConfig.__name__}",
            "AutoModel": f"{flat_name(SpeakUp.__module__)}.{SpeakUp.__name__}",
        },
    )
    model = SpeakUp(config)
    model.objective.load_state_dict(checkpoint_weights(Path(checkpoint), use_ema))
    print(f"Loaded {'EMA' if use_ema else 'raw'} weights from {checkpoint}", flush=True)
    return model.eval(), Path(checkpoint)


def export(model: SpeakUp, folder: Path, card: str) -> None:
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    model.config.save_pretrained(folder)
    save_model(model.objective, str(folder / "model.safetensors"))
    modules = source_closure(SpeakUp.__module__)
    for module in modules:
        (folder / f"{flat_name(module)}.py").write_text(vendored_source(module, modules))
    (folder / "README.md").write_text(card)


def main(run: Path, repo_id: str, dry_run: bool, use_ema: bool) -> None:
    model, checkpoint = load_run(run, use_ema)
    folder = run / "hub"
    export(model, folder, model_card(repo_id, checkpoint, use_ema))
    AutoModel.from_pretrained(str(folder), trust_remote_code=True)
    files = {path.name: path.stat().st_size for path in folder.iterdir()}
    print(f"Saved and reloaded {folder}: {files}", flush=True)
    api = HfApi()
    print(f"Authenticated as {api.whoami()['name']}", flush=True)
    if dry_run:
        print(f"Dry run: would push {sorted(files)} to private repo {repo_id}", flush=True)
        return
    api.create_repo(repo_id, private=True, exist_ok=True)
    commit = api.upload_folder(repo_id=repo_id, folder_path=folder)
    print(f"Pushed to {commit.commit_url}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("repo_id")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--raw", action="store_true")
    args = parser.parse_args()
    main(args.run, args.repo_id, args.dry_run, not args.raw)
