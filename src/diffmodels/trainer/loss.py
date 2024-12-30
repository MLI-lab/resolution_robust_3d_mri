import torch

from torch import Tensor
from src.diffmodels.archs.std.unet import UNetModel
from src.diffmodels import SDE


def epsilon_based_loss_fn(x: Tensor, model: UNetModel, sde: SDE):
    """
    The loss function for training epsilon-based generative models.
    Args:
        x: A mini-batch of training data.
        sde: the forward sde mdoel used.
        model: A PyTorch model instance that represents a
        time-dependent score-based model.
    """

    random_t = torch.randint(1, sde.num_steps, (x.shape[0],), device=x.device)
    z = torch.randn_like(x)
    mean, std = sde.marginal_prob(x, random_t)
    perturbed_x = mean + z * std[:, None, None, None]
    zhat = model(perturbed_x, random_t)
    loss = torch.mean(torch.sum((z - zhat).pow(2), dim=(1, 2, 3)))
    
    return loss
