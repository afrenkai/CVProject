from dataclasses import dataclass
from torch import nn
from architecture.av_model import AudioVisualFlowPredictor
from architecture.pretraining.audio.audio_decoder import CausalAudioDecoder
from architecture.pretraining.audio.encoder import HybridAudioEncoder
from architecture.pretraining.latent_actions import LatentActionModel
from architecture.pretraining.normalization import AudioVisualNormalizer
from architecture.pretraining.recorded_controls import RecordedControlEncoder
from architecture.pretraining.video.encoder import DinoVideoEncoder
from architecture.pretraining.video.video_decoder import DinoVideoDecoder
from training.training_types import ModelModules


@dataclass
class PretrainingModels:
    video_encoder: DinoVideoEncoder
    audio_encoder: HybridAudioEncoder
    audio_decoder: CausalAudioDecoder
    video_decoder: DinoVideoDecoder
    normalizer: AudioVisualNormalizer
    latent_action_model: LatentActionModel
    predictor: AudioVisualFlowPredictor
    control_encoder: RecordedControlEncoder | None = None

    def pretraining_checkpoint_modules(self) -> ModelModules:
        return {name: module for name, module in vars(self).items() if isinstance(module, nn.Module)}
