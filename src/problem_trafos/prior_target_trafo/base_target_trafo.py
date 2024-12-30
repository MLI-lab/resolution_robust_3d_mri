from torch import Tensor
from abc import ABC, abstractmethod

class BaseTargetTrafo(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def __call__(self, x: Tensor) -> Tensor:
        pass
