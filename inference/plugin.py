import json
from pathlib import Path
from sglang.multimodal_gen.registry import register_pipeline
from inference.inference_types import SpeakUpPipelineConfig, SpeakUpSamplingParams
from inference.pipeline import SpeakUpPipeline


def is_speakup_bundle(model_path: str) -> bool:
    if model_path == SpeakUpPipeline.pipeline_name.lower():
        return True
    index = Path(model_path) / "model_index.json"
    return index.is_file() and json.loads(index.read_text()).get("_class_name") == SpeakUpPipeline.pipeline_name


register_pipeline(
    SpeakUpPipeline,
    sampling_param_cls=SpeakUpSamplingParams,
    pipeline_config_cls=SpeakUpPipelineConfig,
    hf_model_paths=["speakup-"],
    model_detectors=[is_speakup_bundle],
)
