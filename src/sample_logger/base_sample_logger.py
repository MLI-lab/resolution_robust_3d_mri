from typing import Any, Dict
from abc import ABC, abstractmethod
from src.representations.base_coord_based_representation import CoordBasedRepresentation

class BaseSampleLogger(ABC):
    """
    Models logging mechanisms for the reconstruction pipeline.
    It is based on the assumption that within a run N samples are reconstruction, with e.g. 1000 iterations per sample.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def init_run(self, **kwargs):
        """
        This method is called once before starting reconstruction.
        """
        pass

    @abstractmethod
    def init_sample_log(self, **kwargs):
        """
        Called once before reconstruction of a sample (but can called multiple times in one run)
        """
        pass

    @abstractmethod
    def __call__(self, representation: CoordBasedRepresentation, step: int, **kwargs):
        """
        Called during reconstruction ()
        """

    @abstractmethod
    def close_sample_log(self, representation: CoordBasedRepresentation):
        """
        Called once after reconstruction of a sample (but can called multiple times in one run)
        """
        pass

    @abstractmethod
    def close_run(self):
        """
        Called once after the reconstruction is finished.
        """
        pass