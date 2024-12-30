from torch import Tensor
from typing import Tuple, Any, Dict, Optional
from src.diffmodels.sde import SDE
from src.problem_trafos.fwd_trafo.base_fwd_trafo import BaseFwdTrafo
from src.problem_trafos.prior_target_trafo.base_prior_trafo import BasePriorTrafo
import logging
import functools

from src.reconstruction.posterior_sampling.impl_linear_cg import linear_cg

import numpy as np
from src.diffmodels.sampler.base_conditioning_method import ConditioningMethod

def decomposed_diffusion_sampling_sde_predictor(
    xhat0 : Tensor,
    rhs: Tensor,
    conj_grad_closure: callable,
    gamma: float,
    alpha: float,
    cg_kwargs: Dict,
) -> Tuple[Tensor, Tensor]:

    xhat_shape = xhat0.shape

    initial_guess = xhat0.reshape(-1, 1)
    rhs_flat = rhs.reshape(-1, 1)

    reg_rhs_flat = rhs_flat * gamma + alpha * initial_guess

    res_before = (rhs_flat - conj_grad_closure(initial_guess)).square().mean()
    xhat, _ = linear_cg(
        matmul_closure=conj_grad_closure,
        rhs=reg_rhs_flat,
        initial_guess=initial_guess,
        **cg_kwargs
    )
    res_after = (rhs_flat - conj_grad_closure(xhat)).square().mean()
    xhat = xhat.T.reshape(xhat_shape)
    # if res_after > res_before + 1e-9:
        # logging.warning(
            # "CG did not converge, res_before: {}, res_after: {}".format(
                # res_before, res_after
            # )
        # )

    return xhat


def conj_grad_closure(
    x: Tensor,
    im_shape: Tuple[int, int],
    fwd_trafo: BaseFwdTrafo,
    gamma: float = 1e-5,
    alpha: float = 1.0,
):
    x = x.T.reshape(1, *im_shape).contiguous()
    return (
        (gamma * fwd_trafo.trafo_adjoint(fwd_trafo(x)) + alpha * x)
        .view(1, np.prod(im_shape))
        .T
    )

class DecomposedDiffusionSampling(ConditioningMethod):

    def __init__(
        self,
        fwd_trafo: BaseFwdTrafo,
        prior_trafo: BasePriorTrafo,
        observation: Tensor,
        sde: SDE,
        im_shape: Tuple[int, int],
        gamma: float = 0.95,
        alpha: float = 1.0,
        cg_max_iter: int = 2,
        cg_max_tridiag_iter: int = 2,
    ):
        super().__init__(fwd_trafo, observation, sde)
        self.prior_trafo = prior_trafo
        self.gamma = gamma
        self.alpha = alpha
        self.cg_kwargs = {
            "max_iter": cg_max_iter,
            "max_tridiag_iter": cg_max_tridiag_iter,
        }  # first >! second

        self.conj_grad_closure_partial = functools.partial(
            conj_grad_closure,
            im_shape=im_shape,
            fwd_trafo=self.fwd_trafo,
            gamma=self.gamma,
            alpha=self.alpha,
        )

    def pre_prediction_step(
        self, x: Tensor, t: Tensor, score_xt: Optional[Tensor], xhat0: Optional[Tensor]
    ) -> Tensor:

        rhs = self.fwd_trafo.trafo_adjoint(self.observation)

        xhat0 = self.prior_trafo.trafo_inv(xhat0)

        xhat = decomposed_diffusion_sampling_sde_predictor(
            rhs=rhs,
            xhat0=xhat0,
            conj_grad_closure=self.conj_grad_closure_partial,
            gamma=self.gamma,
            alpha=self.alpha,
            cg_kwargs=self.cg_kwargs
        )

        xhat = self.prior_trafo(xhat)

        return x, xhat

    def post_prediction_step(
        self, x_pre_cond: Tensor, x_pred: Tensor, t: Tensor, score_xt: Tensor
    ) -> Tensor:

        # don't do any update
        return x_pred
