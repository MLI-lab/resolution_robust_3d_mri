from typing import Optional

import torch

from torch import Tensor

from src.diffmodels.archs.std.unet import UNetModel
from src.diffmodels.sde import SDE
import math
#import einops
from einops import rearrange

def noise_loss(
    output: Tensor,
    outer_iteration : int,
    outer_iterations_max : int,
    score: UNetModel,
    sde: SDE,
    repetition : int = 1,
    reg_strength: float = 1.,
    steps_scaler : float = 0.5,
    time_sampling_method : str = 'random',
    adapt_reg_strength: Optional[bool] = None,
    subsampling_factor: Optional[float] = None, # for DMs which allow subsampling
    ) -> Tensor:

    output = output.repeat(repetition, *[1]*(output.ndim -1))

    if time_sampling_method == 'random':
        t = torch.randint(1, 
            math.floor(steps_scaler * sde.num_steps),
            (output.shape[0],),
            device=output.device
        ) # random time-sampling (allows for batching and single time step reg.)
    elif time_sampling_method == 'linear_descending':
        t = torch.tensor(min(max(
                math.floor(
                    float(steps_scaler) * sde.num_steps * (outer_iterations_max - outer_iteration) / outer_iterations_max  # where 100 is max number of iterations
                ), 0), sde.num_steps - 1), device=output.device).repeat(output.shape[0])
    else:
        raise NotImplementedError(f'time_sampling {time_sampling_method} not implemented')
    
    z = torch.randn_like(output)
    mean, std = sde.marginal_prob(output, t)
    perturbed_x = mean + z * std[:, None, None, None]

    # subsampling
    if subsampling_factor is not None:
        batch_size, channels, height, width = output.shape
        import numpy as np
        size = math.ceil(height * width * subsampling_factor)
        sample_lst = torch.stack(
            [torch.from_numpy(
                    np.random.choice(height*width, size, replace=False)
                ) for _ in range(batch_size)
            ]).to(z.device) # sample lst has shape (B, size), e.g. 50 25600 -> (B, size, 1) -> (B, size, C)

        z = rearrange(z, 'b c h w -> b (h w) c') # x has shape (B, 256*320, 2)
        z = torch.gather(z, dim=1, index=sample_lst.unsqueeze(2).repeat(1,1,channels)).contiguous()

        perturbed_x = rearrange(perturbed_x, 'b c h w -> b (h w) c')
        perturbed_x = torch.gather(perturbed_x, dim=1, index=sample_lst.unsqueeze(2).repeat(1,1,channels)).contiguous()

        zhat = score(perturbed_x, (t, sample_lst, height, width))
    else:
        zhat = score(perturbed_x, t)

    if perturbed_x.size(1) == 1 and zhat.size(1) == 2:
        # this occurs when learn_sigma is enabled for the trained network
        zhat = zhat[:, :1]

    #sum = torch.sum((z - zhat).pow(2), dim=(1,2,3))
    mean = torch.mean((z - zhat).pow(2))

    reg_strength_t = reg_strength
    if adapt_reg_strength: 
        # See Mardani et al. (2023), bar_a is not the bar_a from DDPM's definition here
        bar_a = sde.marginal_prob_mean(t)
        reg_strength_t = std / bar_a * reg_strength

    return mean * reg_strength_t