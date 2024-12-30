from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Union, Tuple, Dict, Any
from torch import Tensor

T = TypeVar('T')

class BaseDatasetTrafo(ABC, Generic[T]):
    def __init__(self,
            provide_pseudoinverse : bool,
            provide_measurement : bool
        ):

        self.provide_pseudoinverse = provide_pseudoinverse
        self.provide_measurement = provide_measurement

    @abstractmethod
    def _transform(
        self, sample: T
    ) -> Tuple[Tensor, Tensor, Tensor, Dict[str, Any]]:
        raise NotImplementedError

    def __call__(self, sample: T) -> Union[Tensor, Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor, Dict[str, Any]]]:
        measurement, target, pseudorec, attrs = self._transform(sample)
        if self.provide_pseudoinverse:
            if self.provide_measurement:
                return measurement, target, pseudorec, attrs
            else:
                return target, pseudorec
        else:
            if self.provide_measurement:
                return measurement, target
            else:
                return target
