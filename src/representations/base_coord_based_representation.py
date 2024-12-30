from typing import Optional, Any, Tuple
from abc import ABC, abstractmethod
from torch import Tensor

from src.representations.mesh import SliceableMesh

class CoordBasedRepresentation(ABC):
    def __init__(self):
        super().__init__()
        # self._cache = None
        # self._update_cntr = 0
        # self._cache_update_cntr = -1

    @abstractmethod
    def forward(self, mesh: SliceableMesh) -> Tensor:
        pass

    @abstractmethod
    def forward_splitted(
        self, mesh: SliceableMesh, custom_device: Optional[Any] = None, split: int = 1
    ) -> Tensor:
        pass

    @abstractmethod
    def get_optimizer_params(self) -> Tuple:
        pass

    # def notify_on_update(self) -> None:
        # self._update_cntr += 1

    # def notify_caching(self) -> None:
        # self._cache_update_cntr = self._update_cntr

    # def can_use_cache(self) -> None:
        # return self._update_cntr == self._cache_update_cntr
