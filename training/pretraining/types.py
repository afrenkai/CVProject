from dataclasses import dataclass
from architecture.av_model import AudioVisualFlowPredictor
from architecture.pretraining.audio.audio_decoder import CausalAudioDecoder
from architecture.pretraining.audio.encoder import HybridAudioEncoder
from architecture.pretraining.latent_actions import LatentActionModel
from architecture.pretraining.normalization import AudioVisualNormalizer
from architecture.pretraining.video.encoder import DinoVideoEncoder
from architecture.pretraining.video.video_decoder import DinoVideoDecoder
from training.types import ModelModules


@dataclass
class PretrainingModels:
    video_encoder: DinoVideoEncoder
    audio_encoder: HybridAudioEncoder
    audio_decoder: CausalAudioDecoder
    video_decoder: DinoVideoDecoder
    normalizer: AudioVisualNormalizer
    latent_action_model: LatentActionModel
    predictor: AudioVisualFlowPredictor

    def pretraining_checkpoint_modules(self) -> ModelModules:
        return {
            "audio_encoder": self.audio_encoder,
            "audio_decoder": self.audio_decoder,
            "video_decoder": self.video_decoder,
            "normalizer": self.normalizer,
            "latent_action_model": self.latent_action_model,
            "predictor": self.predictor,
        }

    def finetuning_checkpoint_modules(self) -> ModelModules:
        return {
            "audio_encoder": self.audio_encoder,
            "latent_action_model": self.latent_action_model,
            "predictor": self.predictor,
            "normalizer": self.normalizer,
        }
