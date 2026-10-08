import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor


@jaxtyped(typechecker=beartype)
def image_reconstruction_loss(
    reconstruction: Float[Tensor, "... channel height width"],
    target: Float[Tensor, "... channel height width"],
) -> Float[Tensor, ""]:
    """
    literally recon loss between a predicted image and a ground truth image sample
    mse, but the nn.MSELoss() returns a callable which we don't want

    @param reconstruction: float tensor with n-ary args but HAS to have channel, width, height
    @param target: float tensor with n-ary args but HAS to have channel, width, height

    Returns: MSE between the recon and target
    """
    return torch.mean((reconstruction - target).square())
