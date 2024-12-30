from typing import Dict
from src.representations.base_coord_based_representation import CoordBasedRepresentation
from .base_sample_logger import BaseSampleLogger
from tqdm import tqdm

class SampleLoggerPass(BaseSampleLogger):

    def __init__(self):
        super().__init__()

    def init_run(self,):
        """
        This method is called once before starting reconstruction.
        """
        pass

    def init_sample_log(self,):
        """
        Called once before sample run (but can called multiple times in one run)
        """
        pass

    def __call__(self, representation: CoordBasedRepresentation, step: int, pbar : tqdm, log_dict : Dict):
        """
        Called during reconstruction or sampling
        """

    def close_sample_log(self, representation: CoordBasedRepresentation):
        """
        Called once after reconstruction of a sample (but can called multiple times in one run)
        """
        pass

    def close_run(self):
        """
        Called once after the reconstruction is finished.
        """
        pass