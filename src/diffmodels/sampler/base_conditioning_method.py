from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

from src.problem_trafos.fwd_trafo.base_fwd_trafo import BaseFwdTrafo
from src.diffmodels.sde import SDE
from torch import Tensor

class ConditioningMethod(ABC):

    def __init__(self, fwd_trafo: BaseFwdTrafo, observation: Tensor, sde: SDE):
        self.fwd_trafo = fwd_trafo
        self.observation = observation
        self.sde = sde

    @abstractmethod
    def pre_prediction_step(
        self,
        x: Tensor,
        t: Tensor,
        score_xt: Optional[Tensor],
        xhat0: Optional[Tensor]
    ) -> Tuple[Tensor, Tensor]:
        """
            Called before the prediction step of the sampler. 

            Returns (updates of) the current state x, and the current MMSE estimate of x0 (xhat0).
        """
        pass

    @abstractmethod
    def post_prediction_step(
        self,
        x_pre_cond : Tensor,
        x_pred: Tensor,
        t: Tensor,
        score_xt: Tensor
    ) -> Tensor:
        """
            Called after the prediction step of the sampler.
            Receives the result after the pre-conditioning step (x_pre_cond) and the result after the subsequent conditioning step (x_pred).

            Returns a conditioned update of the new state x.
        """
        pass
