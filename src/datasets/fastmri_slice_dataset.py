"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import logging
import os
import pickle
import random
import xml.etree.ElementTree as etree
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    NamedTuple,
    Optional,
    Tuple,
    Union,
)
from warnings import warn

import h5py
import numpy as np
import torch
import re

from src.utils.fftn3d import ifftshift, fftshift

from fastmri.data.mri_data import et_query, fetch_dir, FastMRIRawDataSample
from torch import Tensor


class SliceDatasetSample(NamedTuple):
    kspace: Tensor
    target: Tensor
    attrs: Dict[str, Any]


from .base_dataset import BaseDataset


class ExtSliceDataset(BaseDataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    def __init__(
        self,
        root: Union[str, Path, os.PathLike],
        challenge: str,
        transform: Optional[Callable] = None,
        use_dataset_cache: bool = False,
        sample_rate: Optional[float] = None,
        volume_sample_rate: Optional[float] = None,
        dataset_cache_file: Union[str, Path, os.PathLike] = "dataset_cache.pkl",
        num_cols: Optional[Tuple[int]] = None,
        raw_sample_filter: Optional[Callable] = None,
        sensmap_files_root: Optional[str] = None,
        overwrite_sensmap_files: bool = False,
        return_sensmaps: bool = False,
        readout_dim: int = 0,
        readout_dim_is_spatial: bool = True,
        apply_fft1c_on_readout_dim: bool = True,
        apply_fft1c_on_readout_dim_shifts: bool = [True, False],
        dataset_is_3d: bool = True,
        sensemap_3d_slice_from_fixed_view: bool = True,
        perspective_order: Tuple[str, str, str] = ("cor", "sag", "ax"),
        transpose_2D_slice=[False, True, True],
        sensmap_coil_dim_first=True,
        smap_suffix="_sensmap",
        smap_prefix="",
        volume_filter="",
        recons_key="reconstruction_mvue",
        readout_dim_keep_spatial: bool = False,
        sensmaps_key_in_h5: str = "sens_maps",
    ):
        if challenge not in ("singlecoil", "multicoil"):
            raise ValueError('challenge should be either "singlecoil" or "multicoil"')

        if sample_rate is not None and volume_sample_rate is not None:
            raise ValueError(
                "either set sample_rate (sample by slices) or volume_sample_rate (sample by volumes) but not both"
            )

        self.sensmap_files_root = sensmap_files_root
        self.overwrite_sensmap_files = overwrite_sensmap_files
        self.return_sensmaps = return_sensmaps
        self.smap_suffix = smap_suffix
        self.smap_prefix = smap_prefix
        self.root = root

        self.transpose_2D_slice = transpose_2D_slice
        self.sensmap_coil_dim_first = sensmap_coil_dim_first
        self.readout_dim = readout_dim
        self.readout_dim_is_spatial = readout_dim_is_spatial
        self.apply_fft1c_on_readout_dim = apply_fft1c_on_readout_dim
        self.dataset_is_fully_3d = dataset_is_3d
        self.volume_filter = volume_filter

        self.sensemap_3d_slice_from_fixed_view = sensemap_3d_slice_from_fixed_view

        if self.sensmap_files_root is not None:
            if not os.path.exists(self.sensmap_files_root):
                os.makedirs(self.sensmap_files_root)

        self.dataset_cache_file = Path(dataset_cache_file)

        self.perspective_order = perspective_order

        self.transform = transform
        self.recons_key = recons_key
        # self.recons_key = (
        # "reconstruction_mvue"
        ##"reconstruction_esc" if challenge == "singlecoil" else "reconstruction_rss"
        # )
        self.raw_samples = []
        self.raw_sample_filter = raw_sample_filter

        self.num_cols = num_cols
        self.use_dataset_cache = use_dataset_cache

        self.sample_rate = sample_rate
        self.volume_sample_rate = volume_sample_rate

        if self.return_sensmaps:
            self.calc_sensmap_files()
        self._init_dataset()

    def _init_dataset(self):

        # set default sampling mode if none given
        if self.sample_rate is None:
            self.sample_rate = 1.0
        if self.volume_sample_rate is None:
            self.volume_sample_rate = 1.0

        # load dataset cache if we have and user wants to use it
        if self.dataset_cache_file.exists() and self.use_dataset_cache:
            with open(self.dataset_cache_file, "rb") as f:
                dataset_cache = pickle.load(f)
        else:
            dataset_cache = {}

        # check if our dataset is in the cache
        # if there, use that metadata, if not, then regenerate the metadata
        if dataset_cache.get(self.root) is None or not self.use_dataset_cache:
            files = list(Path(self.root).iterdir())
            from tqdm import tqdm

            p = re.compile(self.volume_filter)
            files_filtered = [f for f in files if p.match(f.name)]
            if len(files_filtered) < len(files):
                logging.info(
                    f"Filtered {len(files) - len(files_filtered)} files with filter {self.volume_filter}."
                )
            for fname in tqdm(
                sorted(files_filtered), desc="Retrieving metadata from volumes."
            ):
                metadata, num_slices = self._retrieve_metadata(fname)

                if metadata is None:
                    logging.warning(
                        f"Skipping {fname} due to missing metadata (likely some data missing in the h5 volume)."
                    )
                    continue

                new_raw_samples = []
                for slice_ind in range(num_slices):
                    raw_sample = FastMRIRawDataSample(fname, slice_ind, metadata)
                    if self.raw_sample_filter is None or self.raw_sample_filter(
                        raw_sample, self.root, self.sensmap_files_root
                    ):
                        new_raw_samples.append(raw_sample)

                self.raw_samples += new_raw_samples

            if dataset_cache.get(self.root) is None and self.use_dataset_cache:
                dataset_cache[self.root] = self.raw_samples
                logging.info(f"Saving dataset cache to {self.dataset_cache_file}.")
                with open(self.dataset_cache_file, "wb") as cache_f:
                    pickle.dump(dataset_cache, cache_f)
        else:
            logging.info(f"Using dataset cache from {self.dataset_cache_file}.")
            self.raw_samples = dataset_cache[self.root]

        # subsample if desired
        if self.sample_rate < 1.0:  # sample by slice
            random.shuffle(self.raw_samples)
            num_raw_samples = round(len(self.raw_samples) * self.sample_rate)
            self.raw_samples = self.raw_samples[:num_raw_samples]
        elif self.volume_sample_rate < 1.0:  # sample by volume
            vol_names = sorted(list(set([f[0].stem for f in self.raw_samples])))
            random.shuffle(vol_names)
            num_volumes = round(len(vol_names) * self.volume_sample_rate)
            sampled_vols = vol_names[:num_volumes]
            self.raw_samples = [
                raw_sample
                for raw_sample in self.raw_samples
                if raw_sample[0].stem in sampled_vols
            ]

        if self.num_cols:
            self.raw_samples = [
                ex
                for ex in self.raw_samples
                if ex[2]["encoding_size"][1] in num_cols  # type: ignore
            ]

    def _retrieve_metadata(self, fname):
        with h5py.File(fname, "r") as hf:

            # if "ismrmrd_header" not in hf:
            # warn(f"No ISMRMRD header found in {fname}.")
            # metadata = hf.attrs
            # return metadata, num_slices

            if "ismrmrd_header" in hf:
                et_root = etree.fromstring(hf["ismrmrd_header"][()])

                enc = ["encoding", "encodedSpace", "matrixSize"]
                enc_size = (
                    int(et_query(et_root, enc + ["x"])),
                    int(et_query(et_root, enc + ["y"])),
                    int(et_query(et_root, enc + ["z"])),
                )
                rec = ["encoding", "reconSpace", "matrixSize"]
                recon_size = (
                    int(et_query(et_root, rec + ["x"])),
                    int(et_query(et_root, rec + ["y"])),
                    int(et_query(et_root, rec + ["z"])),
                )

                lims = ["encoding", "encodingLimits", "kspace_encoding_step_1"]
                enc_limits_center = int(et_query(et_root, lims + ["center"]))
                enc_limits_max = int(et_query(et_root, lims + ["maximum"])) + 1

                padding_left = enc_size[1] // 2 - enc_limits_center
                padding_right = padding_left + enc_limits_max
                metadata_ismrmrd = {
                    "padding_left": padding_left,
                    "padding_right": padding_right,
                    "encoding_size": enc_size,
                    "recon_size": recon_size,
                }
            else:
                warn(f"No ISMRMRD header found in {fname}.")
                metadata_ismrmrd = {}

            kspace = hf["kspace"][()]
            num_slices = kspace.shape[0]
            num_coils = kspace.shape[1]

            if self.recons_key not in hf:
                warn(f"Reconstruction key {self.recons_key} not found in {fname}.")
                return None, num_slices

            metadata = {
                "num_slices": num_slices,
                "kspace_vol_norm": np.linalg.norm(kspace),
                # "kspace_vol_std" : np.std(kspace),
                "kspace_shape": kspace.shape,
                "num_coils": num_coils,
                "target_vol_shape": hf[self.recons_key].shape,
                **hf.attrs,
                **metadata_ismrmrd,
            }

            # add here: if we want to cache the sensmaps, we can calculate them here
            # ...

            # set to None to give gc a hint
            kspace = None

        return metadata, num_slices

    def __len__(self):
        return len(self.raw_samples)

    def _fft1c(
        self,
        data: torch.Tensor,
        norm: str = "ortho",
        dim: int = 0,
        shifts_enable: Tuple[bool] = (True, True),
    ) -> torch.Tensor:
        # True, False -> Stanford 3D
        # True, True -> default
        if shifts_enable[0]:
            data = ifftshift(data, dim=[dim])
        data = torch.view_as_real(
            torch.fft.fftn(  # type: ignore
                torch.view_as_complex(data), dim=[dim], norm=norm
            )
        )
        if shifts_enable[1]:
            data = fftshift(data, dim=[dim])
        return data

    def _ifft1c(
        self, data: torch.Tensor, norm: str = "ortho", dim: int = 0
    ) -> torch.Tensor:
        data = ifftshift(data, dim=[dim])
        data = torch.view_as_real(
            torch.fft.ifftn(  # type: ignore
                torch.view_as_complex(data), dim=[dim], norm=norm
            )
        )
        return fftshift(data, dim=[dim])

    def calc_sensmap_files(self):
        from tqdm import tqdm
        from src.problem_trafos.utils.bart_utils import (
            compute_sens_maps_mp,
            compute_sens_maps_3d,
        )

        files = list(Path(self.root).iterdir())
        for fname in sorted(files):

            fname_sensmap = os.path.join(
                str(self.sensmap_files_root),
                self.smap_prefix + str(fname.stem) + self.smap_suffix + ".h5",
            )

            if os.path.exists(fname_sensmap) and not self.overwrite_sensmap_files:
                logging.info(f"Sensmap already present in {fname_sensmap}, skipping")
                continue

            with h5py.File(fname_sensmap, "w") as hf_sens:
                with h5py.File(fname, "r") as hf_data:
                    logging.info(f"Calculating sensmaps for {fname}")

                    kspace = hf_data["kspace"][:]
                    if self.dataset_is_fully_3d:
                        # 3D based sensmaps
                        # kspace shape: (Z, Coils, Y, X, 2), readout_dim is 0
                        kspace = torch.view_as_real(
                            torch.from_numpy(kspace).movedim(1, 0)
                        )
                        if self.readout_dim_is_spatial:
                            kspace = self._fft1c(kspace, dim=1, norm="ortho")
                        # kspace shape is now (Coils, Z, Y, X, 2)
                        sens_maps = compute_sens_maps_3d(kspace)
                    else:
                        sens_maps = compute_sens_maps_mp(kspace)

                    hf_sens.create_dataset("sens_maps", data=sens_maps)

                    if "ismrmrd_header" in hf_data:
                        xml_header = hf_data["ismrmrd_header"][()]
                        hf_sens.create_dataset("ismrmrd_header", data=xml_header)
                    else:
                        warn(f"No ISMRMRD header found in {fname}.")

                    kspace = None

    def __getitem__(self, i: int):
        fname, dataslice, metadata = self.raw_samples[i]

        with h5py.File(fname, "r") as hf:
            kspace = hf["kspace"][dataslice]

            mask = np.asarray(hf["mask"]) if "mask" in hf else None

            target = hf[self.recons_key][dataslice] if self.recons_key in hf else None

            # used for a specific setup
            for po_index, po in enumerate(self.perspective_order):
                if po in self.root and self.transpose_2D_slice[po_index]:
                    kspace = np.moveaxis(kspace, 1, 2)
                    target = np.moveaxis(target, 0, 1)

            attrs = dict(hf.attrs)
            attrs.update(metadata)

            kspace = torch.view_as_real(torch.from_numpy(kspace))
            target = torch.from_numpy(target) if target is not None else None

        if self.return_sensmaps:
            fname_sensmap = os.path.join(
                str(self.sensmap_files_root),
                self.smap_prefix + str(fname.stem) + self.smap_suffix + ".h5",
            )
            with h5py.File(fname_sensmap, "r") as hf_sens:
                # sensmap shape: (Coils, Z, Y, X, 2) if 3d otherwise (Z, Coils, Y, X, 2)

                if self.dataset_is_fully_3d:
                    if self.sensemap_3d_slice_from_fixed_view:

                        if self.sensmap_coil_dim_first:
                            if self.perspective_order[0] in self.root:
                                sens_maps = hf_sens["sens_maps"][:, dataslice]
                            elif self.perspective_order[1] in self.root:
                                sens_maps = hf_sens["sens_maps"][:, :, dataslice]
                            elif self.perspective_order[2] in self.root:
                                sens_maps = hf_sens["sens_maps"][:, :, :, dataslice]
                            else:
                                raise Exception("Unknown orientation")
                        else:
                            if self.perspective_order[0] in self.root:
                                sens_maps = hf_sens["sens_maps"][dataslice]
                            elif self.perspective_order[1] in self.root:
                                sens_maps = hf_sens["sens_maps"][:, dataslice]
                            elif self.perspective_order[2] in self.root:
                                sens_maps = hf_sens["sens_maps"][:, :, dataslice]
                            else:
                                raise Exception("Unknown orientation")
                    else:
                        sens_maps = hf_sens["sens_maps"][:, dataslice]
                else:
                    sens_maps = hf_sens["sens_maps"][dataslice]

                if not self.sensmap_coil_dim_first:
                    sens_maps = np.moveaxis(sens_maps, -1, 0)

                attrs["sens_maps"] = sens_maps  # this is somewhat arbitrary

        sample = SliceDatasetSample(kspace=kspace, target=target, attrs=attrs)

        if self.transform is not None:
            return self.transform(sample)
        else:
            return sample