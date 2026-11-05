import torch
from torch import Tensor, nn
from models.world_model import AudioVisualFlowPredictor


class GuidedPredictor(nn.Module):
    def __init__(self, predictor: AudioVisualFlowPredictor, action_scale: float, context_scale: float) -> None:
        super().__init__()
        self.predictor = predictor
        self.action_scale = action_scale
        self.context_scale = context_scale

    def null_actions(self, actions: Tensor | None) -> Tensor | None:
        if actions is None:
            return None
        return self.predictor.null_action.to(actions.dtype).expand_as(actions)

    def forward(
        self,
        noisy_video: Tensor,
        noisy_audio: Tensor,
        start_time: Tensor,
        end_time: Tensor,
        context_video: Tensor,
        context_audio: Tensor,
        actions: Tensor | None,
        context_actions: Tensor | None = None,
        context_noise: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        video, audio = self.predictor(
            noisy_video,
            noisy_audio,
            start_time,
            end_time,
            context_video,
            context_audio,
            actions,
            context_actions,
            context_noise=context_noise,
        )
        guided_video, guided_audio = video, audio
        if self.action_scale != 1.0:
            free_video, free_audio = self.predictor(
                noisy_video,
                noisy_audio,
                start_time,
                end_time,
                context_video,
                context_audio,
                self.null_actions(actions),
                self.null_actions(context_actions),
                context_noise=context_noise,
            )
            guided_video = guided_video + (self.action_scale - 1.0) * (video - free_video)
            guided_audio = guided_audio + (self.action_scale - 1.0) * (audio - free_audio)
        if self.context_scale != 1.0:
            blind_video, blind_audio = self.predictor(
                noisy_video,
                noisy_audio,
                start_time,
                end_time,
                torch.zeros_like(context_video),
                torch.zeros_like(context_audio),
                actions,
                self.null_actions(context_actions),
                context_noise=context_noise,
            )
            guided_video = guided_video + (self.context_scale - 1.0) * (video - blind_video)
            guided_audio = guided_audio + (self.context_scale - 1.0) * (audio - blind_audio)
        return guided_video, guided_audio
