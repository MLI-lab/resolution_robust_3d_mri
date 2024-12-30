# %%
from typing import Any, Dict, Tuple, Optional
from src.representations.base_coord_based_representation import CoordBasedRepresentation
from src.representations.mesh import SliceableMesh

import wandb
from tqdm import tqdm
from src.utils.wandb_utils import (
    tensor_to_wandbimages_dict
)

import torch

from .base_sample_logger import BaseSampleLogger

class SampleLoggerWithoutTarget(BaseSampleLogger):

    def __init__(self,
            foreach_data: bool,
            final_reco: bool,
            volume_stats_period: int,
            volume_stats__wandb_take_mean_slice_period : int,
            volume_stats__wandb_video_period : int,
            show_phase = bool
        ):
        super().__init__()

        self.foreach_data = foreach_data
        self.final_reco = final_reco
        self.volume_stats_period = volume_stats_period

        self.volume_stats__wandb_take_mean_slice_period = volume_stats__wandb_take_mean_slice_period
        self.volume_stats__wandb_video_period = volume_stats__wandb_video_period
        self.show_phase = show_phase


    def init_run(self,):
        """
        This method is called once before starting reconstruction.
        """
        pass

    def init_sample_log(self, sample_nr : int, mesh : SliceableMesh):
        """
        Called once before sample run (but can called multiple times in one run)
        """
        self.sample_nr = sample_nr
        self.mesh = mesh

    def __call__(self, representation: CoordBasedRepresentation, step: int, pbar : tqdm):
        """
        Called during reconstruction or sampling
        """
        if not self.foreach_data:
            return

        if step % self.volume_stats_period == 0:
            with torch.no_grad():
                sample = representation.forward(self.mesh)

                # generate dict of wandb images
                images = tensor_to_wandbimages_dict(
                    "reco",
                    sample.unsqueeze(0),
                    take_meanslices=step
                    % self.volume_stats__wandb_take_mean_slice_period
                    == 0
                    and step > 0,
                    take_videos=step % self.volume_stats__wandb_video_period == 0
                    and step > 0,
                    show_phase=self.show_phase,
                )

                wandb.log(
                    {
                        "sample_mean": sample.detach().cpu().numpy().mean(),
                        "sample_std": sample.detach().cpu().numpy().std(),
                        "global_step": step,
                        **(images),
                    }
                )

        # old definition from ddim
        # if self.num_img_in_sample_log is not None: 
            # if i % self.num_img_in_sample_log == 0 or i == self.sample_kwargs['num_steps'] - 1:
                # from src.utils.wandb_utils import tensor_to_wandbimage
                # wandb.log(
                    # {
                        # 'reco': tensor_to_wandbimage(x_mean.norm(dim=-3)), 
                        # 'global_step': i,
                    # }
                # )

    def close_sample_log(self, representation: CoordBasedRepresentation):
        """
        Called once after reconstruction of a sample (but can called multiple times in one run)
        """
        if not self.final_reco:
            return
        
        with torch.no_grad():
            sample = representation.forward(self.mesh)

            # generate dict of wandb images
            images = tensor_to_wandbimages_dict(
                "reco",
                sample.unsqueeze(0),
                take_meanslices=False,
                take_videos=True,
                show_phase=self.show_phase,
            )

            wandb.log(
                {
                    "sample_mean": sample.detach().cpu().numpy().mean(),
                    "sample_std": sample.detach().cpu().numpy().std(),
                    "global_step": self.sample_nr,
                    **(images),
                }
            )

    def close_run(self):
        """
        Called once after the reconstruction is finished.
        """
        pass