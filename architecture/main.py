import argparse
from pathlib import Path
import torch
from omegaconf import OmegaConf
from architecture.pretraining.types import WorldModelConfig
from training.pretraining.models import build_flow_predictor


def main() -> None:
    """
    makes gaussians for these based on the size params passed in for demo purposes
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video-config",
        type=Path,
        default=Path("configs/backbones/dinov3.yaml"),
    )
    parser.add_argument(
        "--audio-config",
        type=Path,
        default=Path("configs/backbones/audio_encoder.yaml"),
    )
    parser.add_argument(
        "--world-model-config",
        type=Path,
        default=Path("configs/world_model/default.yaml"),
    )
    arguments = parser.parse_args()

    video_config = OmegaConf.load(arguments.video_config)
    audio_config = OmegaConf.load(arguments.audio_config)
    world_model_config = WorldModelConfig.model_validate(
        OmegaConf.to_container(
            OmegaConf.load(arguments.world_model_config),
            resolve=True,
        )
    )

    device = torch.device("cpu")
    predictor = build_flow_predictor(
        int(video_config.feature_dimension),
        int(audio_config.latent_dimension),
        world_model_config,
        device,
    ).train()

    # W hardcoding
    batch_size = 1
    target_steps = 2
    context_steps = 1
    video_patches = 2

    # v ~ N
    noisy_video = torch.randn(
        batch_size,
        target_steps,
        video_patches,
        int(video_config.feature_dimension),
        device=device,
    )
    # a ~ N
    noisy_audio = torch.randn(
        batch_size,
        target_steps,
        int(audio_config.latent_dimension),
        device=device,
    )

    # ctx_v ~ N
    context_video = torch.randn(
        batch_size,
        context_steps,
        video_patches,
        int(video_config.feature_dimension),
        device=device,
    )

    #ctx_a ~ N
    context_audio = torch.randn(
        batch_size,
        context_steps,
        int(audio_config.latent_dimension),
        device=device,
    )
    # act ~ N
    actions = torch.randn(
        batch_size,
        target_steps,
        world_model_config.action_dimension,
        device=device,
    )

    # could be literally anything, picked these 2
    start_time = torch.full((batch_size,), 0.67, device=device)
    end_time = torch.full((batch_size,), 0.69, device=device)

    #\vdot_v, \vdot_a = P(Enc(vid ~ N(0, I), aud ~ N(0, I), x0,x1 ))
    video_velocity, audio_velocity = predictor(
        noisy_video,
        noisy_audio,
        start_time,
        end_time,
        context_video,
        context_audio,
        actions,
    )
    target_video_velocity = torch.randn_like(video_velocity)
    target_audio_velocity = torch.randn_like(audio_velocity)
    loss = (video_velocity - target_video_velocity).square().mean()
    loss = loss + (audio_velocity - target_audio_velocity).square().mean()
    loss.backward()

    gradients = [
        parameter.grad
        for parameter in predictor.parameters()
        if parameter.grad is not None
    ]
    # implies L2, not wasserstein but normal vector norm is fine as long as all finite. 
    gradient_norm = torch.sqrt(
        torch.stack([gradient.square().sum() for gradient in gradients]).sum()
    )


    print(
        {
            "backbone_width": world_model_config.backbone.hidden_dimension,
            "diffusion_head_width": world_model_config.diffusion_head.hidden_dimension,
            "video_velocity_shape": tuple(video_velocity.shape),
            "audio_velocity_shape": tuple(audio_velocity.shape),
            "outputs_finite": bool(
                torch.isfinite(video_velocity).all()
                and torch.isfinite(audio_velocity).all()
            ),
            "loss": float(loss.detach()),
            "parameters_with_gradients": len(gradients),
            "gradients_finite": all(
                bool(torch.isfinite(gradient).all()) for gradient in gradients
            ),
            "gradient_norm": float(gradient_norm),
        }
    )


if __name__ == "__main__":
    main()
