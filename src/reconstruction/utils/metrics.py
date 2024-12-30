from typing import Optional, Any, Tuple
from torch import Tensor
import torch

from skimage.metrics import structural_similarity

import numpy as np
import scipy
import logging


def PSNR(
    rec: Tensor,
    gt: Tensor,
) -> Tensor:
    return PSNR_pt(reconstruction=rec, ground_truth=gt, data_range=torch.max(gt))


def PSNR_pt(
    reconstruction: Tensor, ground_truth: Tensor, data_range: Optional[Tensor] = None
) -> Tensor:

    mse = (reconstruction - ground_truth).square().mean()
    if data_range is None:
        data_range = torch.max(ground_truth) - np.min(ground_truth)
    return 20 * torch.log10(data_range) - 10 * torch.log10(mse)


def normalize(img):
    img = img - torch.min(img)
    img = img / torch.max(img)
    return img


def PSNR_2D(
    rec: Tensor,
    gt: Tensor,
    axis: int = 0,
    use_vol_max: bool = False,
    take_abs_normalize: bool = False,
) -> Tuple[Tensor, Tensor]:

    if rec.ndim == 5:
        # (B, Z, X, Y, 2)
        rec = rec.squeeze(0)
        gt = gt.squeeze(0)

    if axis != 0:
        rec = rec.moveaxis(axis, 0)
        gt = gt.moveaxis(axis, 0)

    if take_abs_normalize:
        rec = normalize(torch.abs(rec))
        gt = normalize(torch.abs(gt))

    psnrs = PSNR_2D_pt(
        reconstruction=rec,
        ground_truth=gt,
        data_range=torch.max(gt) if use_vol_max else None,
    )

    return psnrs.mean(), psnrs.std()


def PSNR_2D_pt(
    reconstruction: Tensor, ground_truth: Tensor, data_range: Optional[Tensor] = None
) -> Tensor:

    reconstruction = reconstruction.reshape(reconstruction.shape[0], -1)
    ground_truth = ground_truth.reshape(ground_truth.shape[0], -1)

    mse = (reconstruction - ground_truth).square().mean(dim=1)
    if data_range is None:
        data_range = (
            torch.max(ground_truth, dim=1).values
            - torch.min(ground_truth, dim=1).values
        )
    return 20 * torch.log10(data_range) - 10 * torch.log10(mse)


def SSIM(
    rec: Tensor,
    gt: Tensor,
    axis: int = 0,
    max_val: Optional[float] = None,
    take_abs_normalize: bool = False,
) -> Tuple[float, float]:

    if rec.ndim == 4:
        # (B, Z, X, Y)
        rec = rec.squeeze(0)
        gt = gt.squeeze(0)

    if take_abs_normalize:
        rec = normalize(torch.abs(rec))
        gt = normalize(torch.abs(gt))

    if axis != 0:
        rec = rec.moveaxis(axis, 0)
        gt = gt.moveaxis(axis, 0)

    if max_val is None:
        max_val = torch.max(gt).item()

    ssims = ssim_np(
        pred=rec.cpu().reshape(-1, rec.shape[-2], rec.shape[-1]).numpy(),
        gt=gt.cpu().reshape(-1, rec.shape[-2], rec.shape[-1]).numpy(),
        maxval=max_val,
    )

    return ssims.mean(), ssims.std()


def ssim_np(
    gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None
) -> np.ndarray:
    assert gt.ndim == pred.ndim, "Input images must have the same dimensions."
    if gt.ndim == 4 and gt.shape[1] == 1 and pred.shape[1] == 1 and pred.ndim == 4:
        # assume (B, C, H, W)
        gt = gt[:, 0, ...]
        pred = pred[:, 0, ...]
    if not gt.ndim == 3:
        raise ValueError("Unexpected number of dimensions in ground truth.")
    if not gt.ndim == pred.ndim:
        raise ValueError("Ground truth dimensions does not match pred.")

    maxval = gt.max() if maxval is None else maxval

    ssims = np.zeros(gt.shape[0])
    for slice_num in range(gt.shape[0]):
        ssims[slice_num] = structural_similarity(
            gt[slice_num], pred[slice_num], data_range=maxval
        )
    return ssims


def VIFP(
    rec: Tensor,
    gt: Tensor,
) -> np.number:

    return vifp_mscale_np(
        ref=gt.cpu().reshape(-1, gt.shape[-2], gt.shape[-1]).numpy(),
        dist=rec.cpu().reshape(-1, rec.shape[-2], rec.shape[-1]).numpy(),
    )


def vifp_mscale_np(ref, dist, eps=1e-10):
    if not ref.ndim == 3:
        raise ValueError("Unexpected number of dimensions in ground truth.")
    if not ref.ndim == dist.ndim:
        raise ValueError("Ground truth dimensions does not match pred.")

    vifps = np.zeros(ref.shape[0])
    for slice_num in range(ref.shape[0]):
        vifps[slice_num] = vifp_mscale_single_np(
            ref[slice_num][None, ...],
            dist[slice_num][None, ...],
            sigma_nsq=ref[slice_num].mean(),
            eps=eps,
        )

    nr_nans = np.isnan(vifps).sum()
    if nr_nans > 0:
        logging.warn(f"Number of VIFP nans: {nr_nans} / {vifps.size}")

    return np.mean(vifps[~np.isnan(vifps)])


def vifp_mscale_single_np(ref, dist, sigma_nsq=1, eps=1e-10):
    ### from https://github.com/aizvorski/video-quality/blob/master/vifp.py
    sigma_nsq = sigma_nsq  ### tune this for your dataset to get reasonable numbers
    eps = eps

    num = 0.0
    den = 0.0
    for scale in range(1, 5):

        N = 2 ** (4 - scale + 1) + 1
        sd = N / 5.0

        if scale > 1:
            ref = scipy.ndimage.gaussian_filter(ref, sd)
            dist = scipy.ndimage.gaussian_filter(dist, sd)
            ref = ref[::2, ::2]
            dist = dist[::2, ::2]

        mu1 = scipy.ndimage.gaussian_filter(ref, sd)
        mu2 = scipy.ndimage.gaussian_filter(dist, sd)
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = scipy.ndimage.gaussian_filter(ref * ref, sd) - mu1_sq
        sigma2_sq = scipy.ndimage.gaussian_filter(dist * dist, sd) - mu2_sq
        sigma12 = scipy.ndimage.gaussian_filter(ref * dist, sd) - mu1_mu2

        sigma1_sq[sigma1_sq < 0] = 0
        sigma2_sq[sigma2_sq < 0] = 0

        g = sigma12 / (sigma1_sq + eps)
        sv_sq = sigma2_sq - g * sigma12

        g[sigma1_sq < eps] = 0
        sv_sq[sigma1_sq < eps] = sigma2_sq[sigma1_sq < eps]
        sigma1_sq[sigma1_sq < eps] = 0

        g[sigma2_sq < eps] = 0
        sv_sq[sigma2_sq < eps] = 0

        sv_sq[g < 0] = sigma2_sq[g < 0]
        g[g < 0] = 0
        sv_sq[sv_sq <= eps] = eps

        num += np.sum(np.log10(1 + g * g * sigma1_sq / (sv_sq + sigma_nsq)))
        den += np.sum(np.log10(1 + sigma1_sq / sigma_nsq))

    vifp = num / den

    return vifp
