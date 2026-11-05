import argparse
import base64
import io
from pathlib import Path
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from matplotlib import pyplot
from matplotlib.figure import Figure
from data.latent_cache import CachedWindowDataset
from inference.client import SpeakUpClient
from models.encoder_types import FrozenEncoderConfig
from plotting.samples import sample_report
from training.samples import horizon_errors


def figure_png(figure: Figure) -> str:
    """
    makes a figure into b64 encoded and then decoded image (for png)

    @param figure: Figure

    Returns:
        str corresponding 
    """
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=90)
    pyplot.close(figure)
    return base64.b64encode(buffer.getvalue()).decode()

def build_app(client: SpeakUpClient, cache: Path, page: str) -> FastAPI:
    app = FastAPI()
    datasets = {
        directory.name: CachedWindowDataset(directory)
        for directory in sorted(cache.iterdir())
        if any(directory.glob("shard_*.safetensors"))
    }
    contract = client.metadata()
    context_steps = contract["context_steps"]
    timing = FrozenEncoderConfig.model_construct(
        context_frames=contract["context_frames"],
        video_rate=contract["video_rate"],
        video_frames_per_step=contract["video_frames_per_step"],
    )

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return page

    @app.get("/contract")
    def model_contract() -> dict[str, object]:
        return contract | {"datasets": {name: len(windows) for name, windows in datasets.items()}}

    @app.get("/forecast")
    def forecast(
        dataset: str,
        window: int,
        samples: int = 4,
        seed: int = 0,
        steps: str = "1,4,8,16",
        action_guidance: float = 1.0,
        context_guidance: float = 1.0,
    ) -> dict[str, object]:
        windows = datasets[dataset]
        item = windows[window % len(windows)]
        video, audio = item["video"].float(), item["audio"].float()
        observation = {
            "context_video": video[:context_steps].numpy(),
            "context_audio": audio[:context_steps].numpy(),
            "samples_per_candidate": samples,
            "action_guidance": action_guidance,
            "context_guidance": context_guidance,
        }
        if int(item["action_present"]):
            observation["controls"] = item["actions"].numpy()[None]
        results, latencies = {}, {}
        for count in [int(value) for value in steps.split(",")]:
            response = client.forecast(observation, count, seed)
            latencies[count] = response.get("timings", {})
            results[count] = {
                "context_video": video[None, :context_steps],
                "context_audio": audio[None, :context_steps],
                "true_video": video[None, context_steps:],
                "true_audio": audio[None, context_steps:],
                "predicted_video": torch.from_numpy(np.ascontiguousarray(response["video"])),
                "predicted_audio": torch.from_numpy(np.ascontiguousarray(response["audio"])),
            }
        metrics, figures = sample_report(results, None, None, timing)
        return {
            "horizon": {name: values.tolist() for name, values in horizon_errors(results[max(results)]).items()},
            "window": window % len(windows),
            "actions": bool(int(item["action_present"])),
            "metrics": {key.removeprefix("samples/"): round(value, 4) for key, value in metrics.items()},
            "timings": latencies,
        } | {name: figure_png(figure) for name, figure in figures.items()}

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://127.0.0.1:30000")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8000)
    command = parser.parse_args()
    page = (Path(__file__).parent / "webapp.html").read_text()
    uvicorn.run(build_app(SpeakUpClient(command.api), command.cache, page), host="0.0.0.0", port=command.port)
