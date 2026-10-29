from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor
from architecture.pretraining.video.video_decoder import DinoVideoDecoder
from losses.image import image_reconstruction_loss


@jaxtyped(typechecker=beartype)
def video_reconstruction_loss(
    decoder: DinoVideoDecoder,
    latents: Float[Tensor, "batch time patch feature"],
    frames: Float[Tensor, "batch time channel height width"],
) -> Float[Tensor, ""]:
    """
    for a given video, we can decode its latents. however, the decoding can be 
    bad. this can be optimized with a reconstruction loss on its frames. 
    see image.py for the impl (hint, its mse between ground truth and prediction)

    process is we take in latents, pass them into the decoder (which is a pretrained DinoV3 model) which outputs video frames. 

    these frames are a tensor of shape batch x time x channel x height x width, where time 
    corresponds to the current second. 

    We have made the design decision to operate at 30fps, so time is always between 0 and 29 

    actual video frames are also from datasets which are 30fps, so we align them and run this recon loss

    @param decoder: simple decoder which outputs a heatmap from dino latents. dino latents take the form of patches x features over timesteps (see next param)
    @param latents: float tensor of shape batch x time x patch x feature
    @param frames: ground truth float tensor of shape batch x time x channels x height x width which is compared to the output of the decoder(latents)

    Returns (mse(decoder(latents), frames))
    """

    reconstruction = decoder(latents)
    if reconstruction.shape != frames.shape:
        raise ValueError("Video reconstruction and target shapes must match")
    return image_reconstruction_loss(reconstruction, frames)
