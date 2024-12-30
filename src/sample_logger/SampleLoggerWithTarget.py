# %%
from typing import Any, Dict, Tuple, Optional

from abc import ABC, abstractmethod

import torch
from torch import Tensor

from src.representations.base_coord_based_representation import CoordBasedRepresentation
from src.representations.mesh import SliceableMesh
import time

import wandb
from tqdm import tqdm

from src.reconstruction.utils.metrics import PSNR, PSNR_2D, SSIM, VIFP
from src.utils.wandb_utils import tensor_to_wandbimages_dict

import logging

from .base_sample_logger import BaseSampleLogger

class SampleLoggerWithTarget(BaseSampleLogger):
    def __init__(
        self,
        device: str,
        devices: str,
        target_trafo: Any,
        fwd_trafo: Any,
        show_phase: bool,
        sample_gen_split: int,
        volume_stats_period: int,
        medslice_stats_period: int,
        volume_stats__wandb_take_mean_slice_period: int = 5,
        volume_stats__wandb_video_period: int = 300,
        take_abs_normalize: bool = False,
        log_psnr=True,
        log_ssim=False,
        log_vifp=False,
        foreach_data=True,
        final_reco=False,
        save_observation=False,
        save_filtbackproj=False,
        save_ground_truth=False,
        save_final_sample=False,
        store_k3d_volume=False,
        log_gt_fbp_to_wandb=False,
        use_second_cuda=False,
        log_data_is_complex=True,
        log_3d_include_final_slice_averages_for_volumes=False,
    ):

        super().__init__()
        self.device = device
        self.devices = devices
        self.target_trafo = target_trafo
        self.fwd_trafo = fwd_trafo

        self.show_phase = show_phase
        self.sample_gen_split = sample_gen_split
        self.volume_stats_period = volume_stats_period
        self.medslice_stats_period = medslice_stats_period
        self.volume_stats__wandb_take_mean_slice_period = (
            volume_stats__wandb_take_mean_slice_period
        )
        self.volume_stats__wandb_video_period = volume_stats__wandb_video_period
        self.take_abs_normalize = take_abs_normalize
        self.log_psnr = log_psnr
        self.log_ssim = log_ssim
        self.log_vifp = log_vifp

        self.foreach_data = foreach_data
        self.final_reco = final_reco
        self.save_observation = save_observation
        self.save_filtbackproj = save_filtbackproj
        self.save_ground_truth = save_ground_truth
        self.save_final_sample = save_final_sample
        self.store_k3d_volume = store_k3d_volume
        self.log_gt_fbp_to_wandb = log_gt_fbp_to_wandb
        self.use_second_cuda = use_second_cuda
        self.log_3d_include_final_slice_averages_for_volumes = (
            log_3d_include_final_slice_averages_for_volumes
        )
        self.log_data_is_complex = log_data_is_complex

        # sample-dependent
        self.mesh = None
        self.ground_truth = None
        self.scaling_factor = None

    def init_run(self, num_samples: int):
        """
        This method is called once before starting reconstruction.
        """
        if self.use_second_cuda:
            free_devices = {d for d in self.devices if d != self.device}
            assert len(free_devices) > 0, "No free cuda devices available"
            self.eval_device = free_devices[0]
        else:
            self.eval_device = self.device
        logging.info("Using device %s", self.eval_device)

        # create arrays for logging
        if self.log_psnr:
            self.fbp_psnrs = torch.zeros(num_samples, device=self.eval_device)
            self.rec_psnrs = torch.zeros(num_samples, device=self.eval_device)
        if self.log_ssim:
            self.fbp_ssims = torch.zeros(num_samples, device=self.eval_device)
            self.rec_ssims = torch.zeros(num_samples, device=self.eval_device)
        if self.log_vifp:
            self.fbp_vifps = torch.zeros(num_samples, device=self.eval_device)
            self.rec_vifps = torch.zeros(num_samples, device=self.eval_device)
        self.times = torch.zeros(num_samples, device=self.eval_device)

    def init_sample_log(
        self,
        observation: Tensor,
        filtbackproj: Tensor,
        ground_truth: Tensor,
        sample_nr: int,
        scaling_factor: float,
        mesh: SliceableMesh,
    ):

        # save sample-fixed vars for later
        self.observation = observation
        self.ground_truth = ground_truth
        self.scaling_factor = scaling_factor
        self.mesh = mesh

        # optionally saving to disk
        if self.save_observation:
            torch.save(observation, f"observation_{sample_nr}.pt")
        if self.save_filtbackproj:
            torch.save(filtbackproj, f"filtbackproj_{sample_nr}.pt")
        if self.save_ground_truth:
            torch.save(ground_truth, f"ground_truth_{sample_nr}.pt")

        fbp_loss: float = float(
            torch.nn.functional.mse_loss(
                self.fwd_trafo(filtbackproj * scaling_factor),
                observation * scaling_factor,
            )
        )
        trafo_adjoint: Optional[Tensor] = None

        extra_dict = {}
        if self.log_gt_fbp_to_wandb:
            extra_dict = {
                **tensor_to_wandbimages_dict(
                    "ground_truth",
                    ground_truth.unsqueeze(0),
                    take_meanslices=True,
                    take_videos=False,
                    show_phase=self.show_phase,
                ),
                **tensor_to_wandbimages_dict(
                    "fbp",
                    filtbackproj.unsqueeze(0),
                    take_meanslices=True,
                    take_videos=False,
                    show_phase=self.show_phase,
                ),
            }

        trafo_adjoint = self.fwd_trafo.trafo_adjoint(observation)
        if self.log_psnr:
            extra_dict["fbp_psnr"] = PSNR(
                self.target_trafo(trafo_adjoint), ground_truth
            )
        if self.log_ssim:
            extra_dict["fbp_ssim"] = SSIM(
                self.target_trafo(trafo_adjoint),
                ground_truth,
                take_abs_normalize=self.take_abs_normalize,
            )
        if self.log_vifp:
            extra_dict["fbp_vifp"] = VIFP(
                self.target_trafo(trafo_adjoint), ground_truth
            )

        wandb.log(
            {
                "fbp_loss": fbp_loss,
                "scaling_factor": scaling_factor,
                "global_step": sample_nr,
                "step": sample_nr,
                **extra_dict,
            }
        )
        torch.cuda.empty_cache()

        self.sample_nr = sample_nr
        self.time_start = time.time()

    def __call__(
        self,
        representation: CoordBasedRepresentation,
        step: int,
        pbar: tqdm,
        log_dict: Dict = {},
    ):

        if not self.foreach_data:
            return

        if log_dict is not None and len(log_dict) > 0:
            wandb.log({"global_step": step, **log_dict})

        if step % self.volume_stats_period == 0:

            with torch.no_grad():

                sample = representation.forward_splitted(
                    self.mesh, self.eval_device, self.sample_gen_split
                )
                tf_sample = self.target_trafo(sample) / self.scaling_factor
                gt = self.ground_truth.cpu().to(self.eval_device)

                extra_log_dict = {}
                pbar_strs = []
                if self.log_psnr:
                    extra_log_dict["rec_psnr"] = PSNR(tf_sample, gt)
                    pbar_strs.append(f'rec_psnr={extra_log_dict["rec_psnr"]:.1f}')
                if self.log_ssim:
                    tf_sample_ssim = (
                        self.target_trafo(sample).norm(dim=-1)
                        if self.log_data_is_complex
                        else self.target_trafo(sample) / self.scaling_factor
                    )
                    gt_ssim = (
                        self.ground_truth.cpu().norm(dim=-1)
                        if self.log_data_is_complex
                        else self.ground_truth.cpu()
                    )
                    extra_log_dict["ssim_mean"] = SSIM(tf_sample_ssim, gt_ssim)[0]
                    pbar_strs.append(f', rec_ssim={extra_log_dict["ssim_mean"]:.2f}')
                if self.log_vifp:
                    tf_sample_vifp = (
                        self.target_trafo(sample).norm(dim=-1)
                        if self.log_data_is_complex
                        else self.target_trafo(sample) / self.scaling_factor
                    )
                    gt_vifp = (
                        self.ground_truth.cpu().norm(dim=-1)
                        if self.log_data_is_complex
                        else self.ground_truth.cpu()
                    )
                    extra_log_dict["vifp"] = VIFP(tf_sample_vifp, gt_vifp)
                    pbar_strs.append(f', rec_vifp={extra_log_dict["vifp"]:.2f}')

                pbar.set_description(",".join(pbar_strs), refresh=False)

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
                        "rec_mean": sample.detach().cpu().numpy().mean(),
                        "rec_std": sample.detach().cpu().numpy().std(),
                        "global_step": step,
                        **(images),
                        **extra_log_dict,
                    }
                )

        if step % self.medslice_stats_period == 0:

            with torch.no_grad():
                sample_mean_slice = representation.forward(
                    self.mesh.add_index_select(
                        axis=0,
                        indices=torch.Tensor([self.mesh.matrix_size[0] // 2]).int(),
                    )
                )

                extra_log_dict = {}
                pbar_strs = []
                if self.log_psnr:
                    sample_mean_slice_psnr = PSNR_2D(
                        self.target_trafo(sample_mean_slice) / self.scaling_factor,
                        self.ground_truth[self.ground_truth.shape[0] // 2, ...][None]
                        .cpu()
                        .to(self.eval_device),
                        take_abs_normalize=self.take_abs_normalize,
                    )[0]
                    extra_log_dict["rec_medslice_psnr"] = sample_mean_slice_psnr
                    pbar_strs.append(
                        f'rec_medslice_psnr={extra_log_dict["rec_medslice_psnr"]:.1f}'
                    )
                if self.log_ssim:

                    if self.log_ssim:
                        sample_mean_slice_ssim = SSIM(
                            (
                                self.target_trafo(sample_mean_slice).norm(dim=-1)
                                if self.log_data_is_complex
                                else self.target_trafo(sample_mean_slice)
                                / self.scaling_factor
                            ),
                            (
                                self.ground_truth[self.ground_truth.shape[0] // 2, ...]
                                .cpu()
                                .norm(dim=-1)
                                if self.log_data_is_complex
                                else self.ground_truth[
                                    self.ground_truth.shape[0] // 2, ...
                                ].cpu()
                            ),
                            take_abs_normalize=self.take_abs_normalize,
                        )[0]
                        extra_log_dict["rec_medslice_ssim"] = sample_mean_slice_ssim
                        pbar_strs.append(
                            f', rec_medslice_ssim={extra_log_dict["rec_medslice_ssim"]:.2f}'
                        )
                    if self.log_vifp:
                        sample_mean_slice_vifp = VIFP(
                            (
                                self.target_trafo(sample_mean_slice).norm(dim=-1)
                                if self.log_data_is_complex
                                else self.target_trafo(sample_mean_slice)
                                / self.scaling_factor
                            ),
                            (
                                self.ground_truth[self.ground_truth.shape[0] // 2, ...]
                                .cpu()
                                .norm(dim=-1)
                                if self.log_data_is_complex
                                else self.ground_truth[
                                    self.ground_truth.shape[0] // 2, ...
                                ].cpu()
                            ),
                        )
                        extra_log_dict["rec_medslice_vifp"] = sample_mean_slice_vifp
                        pbar_strs.append(
                            f', rec_medslice_vifp={extra_log_dict["rec_medslice_vifp"]:.2f}'
                        )

                pbar.set_description(",".join(pbar_strs), refresh=False)

                wandb.log(
                    {
                        "global_step": step,
                        "rec_medslice_mean": sample_mean_slice.detach()
                        .cpu()
                        .numpy()
                        .mean(),
                        "rec_medslice_std": sample_mean_slice.detach()
                        .cpu()
                        .numpy()
                        .std(),
                        **(
                            tensor_to_wandbimages_dict(
                                "reco_medslice",
                                sample_mean_slice.unsqueeze(0),
                                show_phase=self.show_phase,
                            )
                        ),
                        **extra_log_dict,
                    }
                )

    def close_sample_log(self, representation: CoordBasedRepresentation):
        """
        Called once after reconstruction of a sample (but can called multiple times in one run)
        """
        time_end = time.time()
        self.times[self.sample_nr] = time_end - self.time_start

        sample = representation.forward_splitted(
            self.mesh, self.eval_device, self.sample_gen_split
        )
        tf_sample = self.target_trafo(sample) / self.scaling_factor
        gt = self.ground_truth.cpu().to(self.eval_device)

        if not self.final_reco:
            return

        if self.save_final_sample:
            torch.save(sample, f"final_rec_{self.sample_nr}.pt")

        extra_dict = {}

        if self.log_ssim:
            tf_sample_ssim = (
                tf_sample.norm(dim=-1) if self.log_data_is_complex else tf_sample
            )
            gt_ssim = gt.norm(dim=-1) if self.log_data_is_complex else gt
            self.rec_ssims[self.sample_nr] = SSIM(
                tf_sample_ssim, gt_ssim, take_abs_normalize=self.take_abs_normalize
            )[0]

            extra_dict.update({
                "fbp_ssims_mean": self.fbp_ssims[: self.sample_nr].mean().item(),
                "rec_ssims_mean": self.rec_ssims[: self.sample_nr].mean().item(),
                "rec_ssims_std": self.rec_ssims[: self.sample_nr].std().item(),
            })

        if self.log_vifp:
            tf_sample_vifp = (
                tf_sample.norm(dim=-1) if self.log_data_is_complex else tf_sample
            )
            gt_vifp = gt.norm(dim=-1) if self.log_data_is_complex else gt
            self.rec_vifps[self.sample_nr] = VIFP(tf_sample_vifp, gt_vifp)

            extra_dict.update({
                "fbp_vifps_mean": self.fbp_vifps[: self.sample_nr].mean().item(),
                "rec_vifps_mean": self.rec_vifps[: self.sample_nr].mean().item(),
            })

        if self.log_psnr:
            self.rec_psnrs[self.sample_nr] = PSNR(tf_sample, gt)

            extra_dict.update({
                "fbp_psnrs_mean": self.fbp_psnrs[: self.sample_nr].mean().item(),
                "rec_psnrs_mean": self.rec_psnrs[: self.sample_nr].mean().item(),
                "rec_psnrs_std": self.rec_psnrs[: self.sample_nr].std().item(),
            })

                #"rec_psnr": self.rec_psnrs[self.sample_nr],

        if self.log_3d_include_final_slice_averages_for_volumes:
            with torch.no_grad():
                if self.log_ssim:
                    tf_sample_ssim = (
                        tf_sample.norm(dim=-1)
                        if self.log_data_is_complex
                        else tf_sample
                    )
                    gt_ssim = gt.norm(dim=-1) if self.log_data_is_complex else gt

                    ssim_dim1_mean, ssim_dim1_std = SSIM(
                        tf_sample_ssim,
                        gt_ssim,
                        axis=0,
                        take_abs_normalize=self.take_abs_normalize,
                    )
                    ssim_dim2_mean, ssim_dim2_std = SSIM(
                        tf_sample_ssim,
                        gt_ssim,
                        axis=1,
                        take_abs_normalize=self.take_abs_normalize,
                    )
                    ssim_dim3_mean, ssim_dim3_std = SSIM(
                        tf_sample_ssim,
                        gt_ssim,
                        axis=2,
                        take_abs_normalize=self.take_abs_normalize,
                    )

                    extra_dict.update(
                        {
                            "ssim_dim1_mean": ssim_dim1_mean,
                            "ssim_dim1_std": ssim_dim1_std,
                            "ssim_dim2_mean": ssim_dim2_mean,
                            "ssim_dim2_std": ssim_dim2_std,
                            "ssim_dim3_mean": ssim_dim3_mean,
                            "ssim_dim3_std": ssim_dim3_std,
                        }
                    )

                if self.log_psnr:
                    tf_sample_psnr = (
                        tf_sample.norm(dim=-1)
                        if self.log_data_is_complex
                        else tf_sample
                    )
                    gt_psnr = (
                        gt.norm(dim=-1)
                        if self.log_data_is_complex
                        else gt
                    )

                    psnr_dim1_mean, psnr_dim1_std = PSNR_2D(
                        tf_sample_psnr,
                        gt_psnr,
                        axis=0,
                        use_vol_max=self.log_3d_slice_metrics_use_vol_max,
                        take_abs_normalize=self.take_abs_normalize,
                    )
                    psnr_dim2_mean, psnr_dim2_std = PSNR_2D(
                        tf_sample_psnr,
                        gt_psnr,
                        axis=1,
                        use_vol_max=self.log_3d_slice_metrics_use_vol_max,
                        take_abs_normalize=self.take_abs_normalize,
                    )
                    psnr_dim3_mean, psnr_dim3_std = PSNR_2D(
                        tf_sample_psnr,
                        gt_psnr,
                        axis=2,
                        use_vol_max=self.log_3d_slice_metrics_use_vol_max,
                        take_abs_normalize=self.take_abs_normalize,
                    )

                    extra_dict.update(
                        {
                            "psnr_dim1_mean": psnr_dim1_mean,
                            "psnr_dim1_std": psnr_dim1_std,
                            "psnr_dim2_mean": psnr_dim2_mean,
                            "psnr_dim2_std": psnr_dim2_std,
                            "psnr_dim3_mean": psnr_dim3_mean,
                            "psnr_dim3_std": psnr_dim3_std,
                        }
                    )

        wandb.log(
            {
                "rec_loss": torch.nn.functional.mse_loss(
                    self.fwd_trafo(sample), self.observation
                ),
                "rec_mse": torch.nn.functional.mse_loss(
                    self.target_trafo(sample), self.ground_truth
                ),
                "time": self.times[self.sample_nr],
                "time_mean": self.times[: self.sample_nr].mean().item(),
                "time_std": self.times[: self.sample_nr].std().item(),
                "global_step": self.sample_nr,
                "step": self.sample_nr,
                **extra_dict,
            }
        )

    def close_run(self):
        """
        Called once after the reconstruction is finished.
        """
        if not self.final_reco:
            return

        if wandb.run is not None:
            if self.log_psnr:
                wandb.run.summary["fbp_psnrs_mean"] = self.fbp_psnrs.mean().item()
                wandb.run.summary["fbp_psnrs_std"] = self.fbp_psnrs.std().item()
                wandb.run.summary["rec_psnrs_mean"] = self.rec_psnrs.mean().item()
                wandb.run.summary["rec_psnrs_std"] = self.rec_psnrs.std().item()
            if self.log_ssim:
                wandb.run.summary["fbp_ssims_mean"] = self.fbp_ssims.mean().item()
                wandb.run.summary["fbp_ssims_std"] = self.fbp_ssims.std().item()
                wandb.run.summary["rec_ssims_mean"] = self.rec_ssims.mean().item()
                wandb.run.summary["rec_ssims_std"] = self.rec_ssims.std().item()
            if self.log_vifp:
                wandb.run.summary["fbp_vifps_mean"] = self.fbp_vifps.mean().item()
                wandb.run.summary["fbp_vifps_std"] = self.fbp_vifps.std().item()
                wandb.run.summary["rec_vifps_mean"] = self.rec_vifps.mean().item()
                wandb.run.summary["rec_vifps_std"] = self.rec_vifps.std().item()
