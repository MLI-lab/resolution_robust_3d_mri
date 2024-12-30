import logging
import os
import pickle
import random
import xml.etree.ElementTree as etree
from copy import deepcopy
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)
from warnings import warn

import h5py
import numpy as np
import torch

from src.datasets.base_dataset import BaseDataset

from src.utils.fftn3d import ifftshift, fftshift

from fastmri.data.mri_data import et_query
import re

class FastMRIRawVolumeSample(NamedTuple):
    fname: Path
    slice_count: int
    metadata: Dict[str, Any]

class VolumeDatasetSample(NamedTuple):
    kspace: torch.Tensor
    target: Optional[torch.Tensor]
    attrs: Dict[str, Any]

class FastMRIVolumeDataset(BaseDataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    def __init__(
        self,
        root: Union[str, Path, os.PathLike],
        challenge: str,
        transform: Optional[Callable] = None,
        use_dataset_cache: bool = False,
        volume_sample_rate: Optional[float] = None,
        dataset_cache_file: Union[str, Path, os.PathLike] = "dataset_cache.pkl",
        #num_cols: Optional[Tuple[int]] = None,
        raw_sample_filter: Optional[Callable] = None,
        sensmap_files_root : Optional[str] = None,
        overwrite_sensmap_files : bool = False,
        return_sensmaps : bool = False,
        coil_dim : int = 1, # placeholder
        readout_dim : int = 0,
        readout_dim_is_spatial : bool = True,
        apply_fft1c_on_readout_dim : bool = True,
        apply_fft1c_on_readout_dim_shifts : Tuple[bool] =(True, True),
        readout_dim_keep_spatial : bool = True,
        dataset_is_3d : bool = True,
        volume_filter = "",
        recons_key = "reconstruction_mvue",
        sensmaps_key_in_h5 = "sensitivity_maps",
        sensemap_3d_slice_from_fixed_view : bool = False,
        perspective_order : Tuple[str, str, str] = ("cor", "sag", "ax"),
        transpose_2D_slice : bool = False,
        sensmap_coil_dim_first : bool = False,
        smap_suffix = "_sensmap",
        smap_prefix = "",
    ):
        """
        Args:
            root: Path to the dataset.
            challenge: "singlecoil" or "multicoil" depending on which challenge
                to use.
            transform: Optional; A callable object that pre-processes the raw
                data into appropriate form. The transform function should take
                'kspace', 'target', 'attributes', 'filename', and 'slice' as
                inputs. 'target' may be null for test data.
            use_dataset_cache: Whether to cache dataset metadata. This is very
                useful for large datasets like the brain data.
            sample_rate: Optional; A float between 0 and 1. This controls what fraction
                of the slices should be loaded. Defaults to 1 if no value is given.
                When creating a sampled dataset either set sample_rate (sample by slices)
                or volume_sample_rate (sample by volumes) but not both.
            volume_sample_rate: Optional; A float between 0 and 1. This controls what fraction
                of the volumes should be loaded. Defaults to 1 if no value is given.
                When creating a sampled dataset either set sample_rate (sample by slices)
                or volume_sample_rate (sample by volumes) but not both.
            dataset_cache_file: Optional; A file in which to cache dataset
                information for faster load times.
            num_cols: Optional; If provided, only slices with the desired
                number of columns will be considered.
            raw_sample_filter: Optional; A callable object that takes an raw_sample
                metadata as input and returns a boolean indicating whether the
                raw_sample should be included in the dataset.
        """
        if challenge not in ("singlecoil", "multicoil"):
            raise ValueError('challenge should be either "singlecoil" or "multicoil"')

        if volume_sample_rate is not None and volume_sample_rate is not None:
            raise ValueError(
                "either set sample_rate (sample by slices) or volume_sample_rate (sample by volumes) but not both"
            )

        self.sensmap_files_root = sensmap_files_root
        self.overwrite_sensmap_files = overwrite_sensmap_files
        self.return_sensmaps = return_sensmaps
        self.root = root

        self.coil_dim = coil_dim
        self.readout_dim = readout_dim
        self.readout_dim_is_spatial = readout_dim_is_spatial
        self.readout_dim_keep_spatial = readout_dim_keep_spatial
        self.apply_fft1c_on_readout_dim = apply_fft1c_on_readout_dim
        self.apply_fft1c_on_readout_dim_shifts = apply_fft1c_on_readout_dim_shifts
        self.dataset_is_fully_3d = dataset_is_3d
        self.volume_filter = volume_filter
        self.smap_suffix = smap_suffix
        self.smap_prefix = smap_prefix

        self.sensmaps_key_in_h5 = sensmaps_key_in_h5

        if self.sensmap_files_root is not None:
            if not os.path.exists(self.sensmap_files_root):
                os.makedirs(self.sensmap_files_root)

        self.dataset_cache_file = Path(dataset_cache_file)

        self.transform = transform
        self.recons_key = recons_key
        # self.recons_key = (
        ## "reconstruction_esc" if challenge == "singlecoil" else "reconstruction_rss"
        # "reconstruction_mvue"
        # )
        self.raw_samples = []
        if raw_sample_filter is None:
            self.raw_sample_filter = lambda raw_sample: True
        else:
            self.raw_sample_filter = raw_sample_filter

        # set default sampling mode if none given
        if volume_sample_rate is None:
            volume_sample_rate = 1.0

        # load dataset cache if we have and user wants to use it
        if self.dataset_cache_file.exists() and use_dataset_cache:
            with open(self.dataset_cache_file, "rb") as f:
                dataset_cache = pickle.load(f)
        else:
            dataset_cache = {}

        # check if our dataset is in the cache
        # if there, use that metadata, if not, then regenerate the metadata
        if dataset_cache.get(root) is None or not use_dataset_cache:
            files = list(Path(root).iterdir())
            p = re.compile(self.volume_filter)
            files_filtered = [f for f in files if p.match(f.name)]
            if len(files_filtered) < len(files):
                logging.info(f"Filtered {len(files) - len(files_filtered)} files with filter {self.volume_filter}.")
            for fname in sorted(files_filtered):
                metadata, num_slices = self._retrieve_metadata(fname)

                raw_sample = FastMRIRawVolumeSample(fname, num_slices, metadata)

                if raw_sample_filter is None or raw_sample_filter(raw_sample, root, sensmap_files_root):
                    self.raw_samples.append(raw_sample)

            if dataset_cache.get(root) is None and use_dataset_cache:
                dataset_cache[root] = self.raw_samples
                logging.info(f"Saving dataset cache to {self.dataset_cache_file}.")
                with open(self.dataset_cache_file, "wb") as cache_f:
                    pickle.dump(dataset_cache, cache_f)
        else:
            logging.info(f"Using dataset cache from {self.dataset_cache_file}.")
            self.raw_samples = dataset_cache[root]

        # subsample if desired
        if volume_sample_rate < 1.0:
            random.shuffle(self.raw_samples)
            num_raw_samples = round(len(self.raw_samples) * volume_sample_rate)
            self.raw_samples = self.raw_samples[:num_raw_samples]

    def _retrieve_metadata(self, fname):
        with h5py.File(fname, "r") as hf:

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
                    "recon_size": recon_size
                }
            else:
                warn(f"No ISMRMRD header found in {fname}.")
                metadata_ismrmrd = {}

            num_slices = hf["kspace"].shape[0]

            metadata = {
                "num_slices" : num_slices,
                **hf.attrs,
                **metadata_ismrmrd
            }

        return metadata, num_slices

    def calc_sensmap_files(self):
        if self.sensmaps_key_in_h5 is not None:
            logging.info(f"Loading sensmaps from the h5 files, skipping calculation.")
            return

        from tqdm import tqdm
        from src.problem_trafos.utils.bart_utils import compute_sens_maps_mp, compute_sens_maps_3d

        files = list(Path(self.root).iterdir())
        for fname in sorted(files):

            fname_sensmap = os.path.join(str(self.sensmap_files_root), self.smap_prefix + str(fname.stem) + self.smap_suffix + ".h5")

            if os.path.exists(fname_sensmap) and not self.overwrite_sensmap_files:
                logging.info(f"Sensmap already present in {fname_sensmap}, skipping")
                continue

            with h5py.File(fname_sensmap, 'w') as hf_sens:
                with h5py.File(fname, "r") as hf_data:
                    logging.info(f"Calculating sensmaps for {fname}")

                    kspace = hf_data["kspace"][:]
                    # standard shape is: (Z, Coils, Y, X, 2)

                    if self.dataset_is_fully_3d:
                        # 3D based sensmaps
                        # kspace shape: (Z, Coils, Y, X, 2), readout_dim is 0
                        kspace = torch.view_as_real(torch.from_numpy(kspace).movedim(1,0))
                        if self.readout_dim_is_spatial:
                            kspace = self._fft1c(kspace, dim=1, norm="ortho")
                        # kspace shape is now (Coils, Z, Y, X, 2)
                        sens_maps = compute_sens_maps_3d(kspace)
                    else:
                        # 2D based sensmaps
                        # assume shape: (Z, Coils, Y, X, 2)
                        sens_maps = compute_sens_maps_mp(kspace)

                    hf_sens.create_dataset('sens_maps', data=sens_maps)

                    if "ismrmrd_header" in hf_data:
                        xml_header = hf_data['ismrmrd_header'][()]
                        hf_sens.create_dataset('ismrmrd_header', data=xml_header)
                    else:
                        warn(f"No ISMRMRD header found in {fname}.")

                    kspace = None # delete reference

    def __len__(self):
        return len(self.raw_samples)

    def _fft1c(self, data: torch.Tensor, norm: str = "ortho", dim:int = 0, shifts_enable : Tuple[bool] = (True, True)) -> torch.Tensor:
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

    def _ifft1c(self, data: torch.Tensor, norm: str = "ortho", dim:int = 0) -> torch.Tensor:
        data = ifftshift(data, dim=[dim])
        data = torch.view_as_real(
            torch.fft.ifftn(  # type: ignore
                torch.view_as_complex(data), dim=[dim], norm=norm
            )
        )
        return fftshift(data, dim=[dim])

    def __getitem__(self, i: int):
        fname, slice_count, metadata = self.raw_samples[i]

        with h5py.File(fname, "r") as hf:
            kspace = hf["kspace"][:] #[dataslice]

            kspace = torch.view_as_real(torch.from_numpy(kspace))
            if self.dataset_is_fully_3d:
                kspace = kspace.movedim(1, 0)

                if self.apply_fft1c_on_readout_dim:
                    kspace = self._fft1c(kspace, dim=1, norm="ortho", shifts_enable=self.apply_fft1c_on_readout_dim_shifts)
                    if self.readout_dim_keep_spatial:
                        kspace = self._ifft1c(kspace, dim=1, norm="ortho")
                        # if both enabled this effectively performs the z-dir readout kspace correction

            mask = np.asarray(hf["mask"]) if "mask" in hf else None

            target = hf[self.recons_key][:] if self.recons_key in hf else None

            target = torch.from_numpy(target)

            attrs = dict(hf.attrs)

            if self.return_sensmaps and self.sensmaps_key_in_h5 is not None:
                attrs["sens_maps"] = np.moveaxis(hf[self.sensmaps_key_in_h5][:], -3,-1)

            attrs.update(metadata)

        if self.return_sensmaps and self.sensmaps_key_in_h5 is None:
            fname_sensmap = os.path.join(str(self.sensmap_files_root), self.smap_prefix + str(fname.stem) + self.smap_suffix + ".h5")
            with h5py.File(fname_sensmap, "r") as hf_sens:
                attrs["sens_maps"] = hf_sens["sens_maps"][:]

        sample = VolumeDatasetSample(kspace=kspace, target=target, attrs=attrs)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
