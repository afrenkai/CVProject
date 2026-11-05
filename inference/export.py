import argparse
import json
import shutil
from pathlib import Path
from safetensors.torch import save_file
from training.objective import checkpoint_weights


def export_bundle(run: Path, checkpoint: Path, output: Path, use_ema: bool) -> None:
    output.mkdir(parents=True, exist_ok=True)
    state = checkpoint_weights(checkpoint, use_ema)
    save_file({name: tensor.contiguous() for name, tensor in state.items()}, output / "model.safetensors")
    shutil.copy(run / "pretrain_config.json", output / "pretrain_config.json")
    shutil.copy(run / "normalizer.pt", output / "normalizer.pt")
    index = {
        "_class_name": "SpeakUpPipeline",
        "_diffusers_version": "0.37.0",
        "source_checkpoint": str(checkpoint),
        "weights": "ema" if use_ema else "raw",
    }
    (output / "model_index.json").write_text(json.dumps(index, indent=2))
    print(f"Exported {index['weights']} weights from {checkpoint} to {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--raw", action="store_true")
    command = parser.parse_args()
    export_bundle(command.run, command.checkpoint, command.output, not command.raw)
