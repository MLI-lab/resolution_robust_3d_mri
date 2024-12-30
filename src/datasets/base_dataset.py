from abc import ABC, abstractmethod

from torch.utils.data import Dataset
from torch import Tensor

class BaseDataset(ABC, Dataset[Tensor]):
    def __init__(self,
        ):
        pass

    @abstractmethod
    def __len__(self):
        raise NotImplementedError