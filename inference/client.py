import argparse
import time
from pathlib import Path
import httpx
import msgpack
import numpy as np
from data.latent_cache import CachedWindowDataset


def pack_array(value: object) -> object:
    if isinstance(value, np.ndarray):
        return {"__ndarray__": True, "data": value.tobytes(), "dtype": value.dtype.str, "shape": list(value.shape)}
    return value


def unpack_array(value: dict[str | bytes, object]) -> object:
    fields = {key.decode() if isinstance(key, bytes) else key: item for key, item in value.items()}
    if fields.get("__ndarray__"):
        return np.frombuffer(fields["data"], dtype=np.dtype(fields["dtype"])).reshape(fields["shape"])
    return value


def bfloat16_to_float32(bits: np.ndarray) -> np.ndarray:
    return (bits.view(np.uint16).astype(np.uint32) << 16).view(np.float32)


class SpeakUpClient:
    def __init__(self, url: str, timeout: float = 600.0) -> None:
        self.http = httpx.Client(base_url=url, timeout=timeout)

    def metadata(self) -> dict[str, object]:
        response = self.http.get("/v1/actions/metadata")
        response.raise_for_status()
        return response.json()

    def forecast(self, observation: dict[str, object], num_inference_steps: int, seed: int) -> dict[str, object]:
        payload = {
            "input": {"observation": observation},
            "parameters": {"num_inference_steps": num_inference_steps, "seed": seed},
            "runtime": {"response_format": "raw", "output_format": "numpy"},
        }
        response = self.http.post(
            "/v1/actions/generations",
            content=msgpack.packb(payload, default=pack_array, use_bin_type=True),
            headers={"content-type": "application/msgpack", "accept": "application/msgpack"},
        )
        if response.status_code != 200:
            raise RuntimeError(f"{response.status_code}: {response.text[:2000]}")
        result = msgpack.unpackb(response.content, object_hook=unpack_array, raw=False)
        if result["dtype"] == "bfloat16":
            result["video"] = bfloat16_to_float32(result["video"])
            result["audio"] = bfloat16_to_float32(result["audio"])
        return result


def check_cached_window(
    client: SpeakUpClient, cache: Path, window: int, samples: int, steps: int, candidates: int
) -> None:
    contract = client.metadata()
    context_steps = contract["context_steps"]
    item = CachedWindowDataset(cache)[window]
    video, audio = item["video"].float().numpy(), item["audio"].float().numpy()
    controls = np.repeat(item["actions"].numpy()[None], candidates, axis=0)
    observation = {
        "context_video": video[:context_steps],
        "context_audio": audio[:context_steps],
        "samples_per_candidate": samples,
    }
    if int(item["action_present"]):
        observation["controls"] = controls
    start = time.perf_counter()
    result = client.forecast(observation, steps, seed=window)
    latency = time.perf_counter() - start
    true_video, true_audio = video[context_steps:], audio[context_steps:]
    copy_video = np.broadcast_to(video[context_steps - 1 : context_steps], true_video.shape)
    copy_audio = np.broadcast_to(audio[context_steps - 1 : context_steps], true_audio.shape)
    video_ratio = np.square(result["video"] - true_video).mean() / np.square(copy_video - true_video).mean()
    audio_ratio = np.square(result["audio"] - true_audio).mean() / np.square(copy_audio - true_audio).mean()
    print(f"contract {contract}", flush=True)
    print(f"video {list(result['video'].shape)} audio {list(result['audio'].shape)} latency {latency:.2f}s", flush=True)
    print(f"server timings {result.get('timings')}", flush=True)
    print(f"MSE / copy-last: video {video_ratio:.3f} audio {audio_ratio:.3f}", flush=True)


def benchmark(client: SpeakUpClient, cache: Path, repeats: int, first_row_only: bool) -> None:
    contract = client.metadata()
    context_steps = contract["context_steps"]
    item = CachedWindowDataset(cache)[0]
    video, audio = item["video"].float().numpy(), item["audio"].float().numpy()
    controls = item["actions"].numpy()[None]
    print("rows  nfe  server_denoise_ms  round_trip_ms", flush=True)
    for rows in (1, 4, 16, 64):
        for steps in (1, 4, 16):
            observation = {
                "context_video": video[:context_steps],
                "context_audio": audio[:context_steps],
                "controls": controls,
                "samples_per_candidate": rows,
            }
            if first_row_only:
                observation["return_rows"] = [0]
            denoise, round_trip = [], []
            for repeat in range(repeats + 1):
                start = time.perf_counter()
                result = client.forecast(observation, steps, seed=repeat)
                if repeat:
                    round_trip.append(1000 * (time.perf_counter() - start))
                    denoise.append(result["timings"]["denoise_ms"])
            print(f"{rows:4d} {steps:4d} {np.median(denoise):18.1f} {np.median(round_trip):14.1f}", flush=True)


def check_raw_context(client: SpeakUpClient, steps: int) -> None:
    contract = client.metadata()
    generator = np.random.default_rng(0)
    observation = {
        "frames": generator.integers(0, 256, (contract["context_frames"], 360, 640, 3), dtype=np.uint8),
        "waveform": 0.1
        * generator.standard_normal((contract["waveform_channels"], contract["context_samples"]), dtype=np.float32),
        "samples_per_candidate": 2,
    }
    result = client.forecast(observation, steps, seed=0)
    print(f"raw context: video {list(result['video'].shape)} audio {list(result['audio'].shape)}", flush=True)
    print(f"server timings {result.get('timings')}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:30000")
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--window", type=int, default=0)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--benchmark", type=int, default=0)
    parser.add_argument("--first-row-only", action="store_true")
    command = parser.parse_args()
    client = SpeakUpClient(command.url)
    if command.benchmark:
        benchmark(client, command.cache, command.benchmark, command.first_row_only)
    elif command.cache is not None:
        check_cached_window(client, command.cache, command.window, command.samples, command.steps, command.candidates)
    if not command.benchmark:
        check_raw_context(client, command.steps)
