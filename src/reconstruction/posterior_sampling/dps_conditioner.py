
import torch
from torch import Tensor
from typing import Tuple, Any, Dict, Optional
from src.diffmodels.sde import SDE
from src.problem_trafos.fwd_trafo.base_fwd_trafo import BaseFwdTrafo
from src.diffmodels.sampler.base_conditioning_method import ConditioningMethod

class PosteriorSampling(ConditioningMethod):

    def __init__(
        self,
        fwd_trafo: BaseFwdTrafo,
        prior_trafo: BaseFwdTrafo,
        observation: Tensor,
        sde: SDE,
        scale: float = 0.01,
        score_derivative: bool = False,
    ):
        super().__init__(fwd_trafo, observation, sde)
        self.prior_trafo = prior_trafo
        self.scale = scale
        self.score_derivative = score_derivative

    def pre_prediction_step(
        self,
        x: Tensor,
        t: Tensor,
        score_xt: Optional[Tensor],
        xhat0: Optional[Tensor]
    ) -> Tuple[Tensor, Tensor]:
        """
            Don't update.
        """
        return x, xhat0

    def post_prediction_step(
        self,
        x_pre_cond : Tensor,
        x_pred: Tensor,
        t: Tensor,
        score_xt: Tensor
    ) -> Tensor:

        x_0_hat = self.sde.posterior_mean_approx(x_pred, t, score_xt)
        x_0_hat_transf = self.prior_trafo.trafo_inv(x_0_hat) if self.prior_trafo is not None else x_0_hat

        difference = self.observation - self.fwd_trafo.trafo(x_0_hat_transf.contiguous())
        norm = difference.square().mean()

        norm_grad = torch.autograd.grad(outputs=norm, inputs=
            x_0_hat if not self.score_derivative else x_pre_cond)[0]

        # x_new -= norm_grad * self.scale
        return x_pred - norm_grad * self.scale