# %%
from typing import Any, Dict, Optional, Tuple, List

import torch
from torch import Tensor

from src.problem_trafos.prior_target_trafo.base_prior_trafo import BasePriorTrafo
from src.problem_trafos.fwd_trafo.base_fwd_trafo import BaseFwdTrafo

from src.diffmodels.sde import SDE
from src.diffmodels.archs.std.unet import UNetModel
from src.reconstruction.posterior_sampling.conditioner_resolver import (
    ConditioningMethod,
)

from .base_sampler import BaseSampler

class DDPM(BaseSampler):
    def __init__(
        self,
        im_shape,
        sampling_in_3d : bool,
        score_mini_batch_size : int,
        cycling_skip_conditioning : bool,
        cycling : bool,
        num_steps : int,
        score: UNetModel,
        sde: SDE,
        device: Optional[str] = None,
        conditioning_method: Optional[ConditioningMethod] = None,
        fwd_trafo: Optional[BaseFwdTrafo] = None,
        prior_trafo: Optional[BasePriorTrafo] = None,
        sample_logger: Optional[Any] = None
    ):
        super().__init__(
            score=score,
            sde=sde,
            device=device,
            conditioning_method=conditioning_method,
            fwd_trafo=fwd_trafo,
            prior_trafo=prior_trafo,
            sample_logger=sample_logger,
            sampling_in_3d = sampling_in_3d,
            im_shape=im_shape,
            score_mini_batch_size=score_mini_batch_size,
            cycling_skip_conditioning=cycling_skip_conditioning,
            cycling=cycling,
        )
        self.num_steps = num_steps

    def _init_timeschedule(self) -> List[Tuple[int, int]]:
        assert self.sde.num_steps == self.num_steps
        timesteps = range(self.num_steps - 1, -2, -1)
        return list(zip(timesteps[:-1], timesteps[1:]))

    def _predictor(
        self,
        score_xt: Tensor,
        x: Tensor,
        t: Tuple[Tensor, Tensor],
        xhat0: Optional[Tensor],
        sde: SDE,
    ) -> Tuple[Tensor, Tensor]:

        current_time, previous_time = t  # current_time > previous_time
        mean_prev_time = sde.marginal_prob_mean(t=previous_time)[:, None, None, None]
        mean_curr_time = sde.marginal_prob_mean(t=current_time)[:, None, None, None]

        alpha_t = mean_curr_time.pow(2) / mean_prev_time.pow(2)
        alpha_bar_t = mean_curr_time.pow(2)

        # DDPM paper, lower bound
        sqrt_beta = (
            (1 - mean_prev_time.pow(2)) / (1 - mean_curr_time.pow(2))
        ).sqrt() * (1 - alpha_t).sqrt()
        # DDPM paper, upper bound
        # sqrt_beta = torch.sqrt(1.0-alpha_t)
        xmean = (
            x - (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t) * score_xt
        ) / torch.sqrt(alpha_t)
        x = xmean + sqrt_beta * torch.randn_like(x)
        return x.detach(), xmean.detach()
