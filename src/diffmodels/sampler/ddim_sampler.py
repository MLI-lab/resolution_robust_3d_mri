# %%
from abc import ABC, abstractmethod
from torch import Tensor
from typing import Any, Dict, Optional, Tuple, List

import torch

from torch import Tensor
from src.problem_trafos.prior_target_trafo.base_prior_trafo import BasePriorTrafo
from src.problem_trafos.fwd_trafo.base_fwd_trafo import BaseFwdTrafo

from src.diffmodels.sde import SDE
from src.diffmodels.archs.std.unet import UNetModel
from src.reconstruction.posterior_sampling.conditioner_resolver import ConditioningMethod

from .base_sampler import BaseSampler

# %%
def _schedule_jump(num_steps: int, travel_length: int = 1, travel_repeat: int = 1):
    jumps = {}
    for j in range(0, num_steps - travel_length, travel_length):
        jumps[j] = travel_repeat - 1

    t = num_steps
    time_steps = []
    while t >= 1:
        t = t - 1
        time_steps.append(t)
        if jumps.get(t, 0) > 0:
            jumps[t] = jumps[t] - 1
            for _ in range(travel_length):
                t = t + 1
                time_steps.append(t)
    time_steps.append(-1)

    return time_steps

def _ddim(
    s: Tensor,
    xhat: Tensor,
    ts: Tuple[Tensor, Tensor],
    sde: SDE,
    eta: float
    ) -> Tensor:

    current_time, previous_time = ts
    mean_prev_time = sde.marginal_prob_mean(
            t=previous_time)[:, None, None, None]
    mean_curr_time = sde.marginal_prob_mean(
            t=current_time)[:, None, None, None]
    
    sqrt_beta = ((1 - mean_prev_time.pow(2)) / (1 - mean_curr_time.pow(2))).sqrt() * \
                (1 - mean_curr_time.pow(2) / mean_prev_time.pow(2)).sqrt()
    if sqrt_beta.isnan().any():
        sqrt_beta = torch.zeros_like(sqrt_beta, device= s.device)
    scaled_noise = xhat * mean_prev_time
    deterministic_noise = torch.sqrt(1 - mean_prev_time.pow(2) - sqrt_beta.pow(2) * eta**2) *  s
    stochastic_noise = eta * sqrt_beta * torch.randn_like(xhat)

    return scaled_noise + deterministic_noise + stochastic_noise

class DDIM(BaseSampler):
    def __init__(self,
            sampling_in_3d : bool,
            im_shape,
            score_mini_batch_size : int,
            cycling_skip_conditioning : bool,
            cycling : bool,
            num_steps : int,
            eta : float,
            score: UNetModel,
            sde: SDE,
            device: Optional[Any] = None,
            conditioning_method : Optional[ConditioningMethod] = None,
            fwd_trafo : Optional[BaseFwdTrafo] = None,
            prior_trafo : Optional[BasePriorTrafo] = None,
            sample_logger : Optional[Any] = None
        ):
        super().__init__(
            score=score,
            sde=sde,
            device=device,
            conditioning_method=conditioning_method,
            fwd_trafo=fwd_trafo,
            prior_trafo=prior_trafo,
            sample_logger=sample_logger,
            sampling_in_3d=sampling_in_3d,
            im_shape=im_shape,
            score_mini_batch_size=score_mini_batch_size,
            cycling_skip_conditioning=cycling_skip_conditioning,
            cycling=cycling,
        )
        self.num_steps = num_steps
        self.eta = eta

    def _init_timeschedule(self) -> List[Tuple[int, int]]:
        assert self.sde.num_steps >= self.num_steps
        skip = self.sde.num_steps // self.num_steps

        ts = _schedule_jump(self.num_steps)
        time_pairs = list(
            (i * skip , j * skip if j > 0 else -1)
            for i, j in zip(ts[:-1], ts[1:])
        )        
        return time_pairs

    def _predictor(self,
            score_xt: Tensor,
            x: Tensor,
            xhat0 : Optional[Tensor],
            t: Tuple[Tensor, Tensor],
            ) -> Tuple[Tensor, Tensor]:

        with torch.no_grad():
            s = score_xt.detach()
            if xhat0 is None:
                xhat0 = self.sde.tweedy(
                    x = x,
                    t = t[0],
                    score_xt = s)
            x = _ddim(
                s=s,
                xhat=xhat0,
                ts=t,
                sde=self.sde,
                eta=self.eta
                )

        return x.detach(), xhat0.detach()