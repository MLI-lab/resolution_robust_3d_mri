"""
    Provides data and experimental utilities.
"""

from typing import Any

from src.problem_trafos.fwd_trafo.base_fwd_trafo import BaseFwdTrafo
from src.problem_trafos.prior_target_trafo.base_target_trafo import BaseTargetTrafo
from src.problem_trafos.prior_target_trafo.base_prior_trafo import BasePriorTrafo
from src.problem_trafos.dataset_trafo.base_dataset_trafo import BaseDatasetTrafo


def get_fwd_trafo(name: str, **cfg_kwargs) -> BaseFwdTrafo:
    if name in ("mri3d"):
        from src.problem_trafos.fwd_trafo.mri_3d_trafo import SubsampledFourierTrafo3D

        fwd_trafo = SubsampledFourierTrafo3D(**cfg_kwargs)

    elif name in ("identity"):
        from src.problem_trafos.fwd_trafo.identity import IdentityTrafo

        fwd_trafo = IdentityTrafo()
    else:
        raise ValueError

    return fwd_trafo


def get_target_trafo(name: str, **cfg_kwargs) -> BaseTargetTrafo:
    from src.problem_trafos.prior_target_trafo.CroppedMagnitudeImageTrafo import (
        CroppedMagnitudeImagePriorTrafo,
    )
    from src.problem_trafos.prior_target_trafo.IdentityTrafo import IdentityTrafo

    if name in (CroppedMagnitudeImagePriorTrafo.__name__):
        return CroppedMagnitudeImagePriorTrafo(**cfg_kwargs)
    elif name in (IdentityTrafo.__name__):
        return IdentityTrafo()
    else:
        raise NotImplementedError


def get_prior_trafo(name: str, **cfg_kwargs) -> BasePriorTrafo:
    from src.problem_trafos.prior_target_trafo.CroppedMagnitudeImageTrafo import (
        CroppedMagnitudeImagePriorTrafo,
    )
    from src.problem_trafos.prior_target_trafo.IdentityTrafo import IdentityTrafo

    if name in (CroppedMagnitudeImagePriorTrafo.__name__):
        return CroppedMagnitudeImagePriorTrafo(**cfg_kwargs)
    elif name in (IdentityTrafo.__name__):
        return IdentityTrafo()
    else:
        raise NotImplementedError


def get_dataset_trafo(
    name: str,
    device: str = "cpu",
    provide_pseudoinverse: bool = False,
    provide_measurement: bool = True,
    **cfg_kwargs
) -> BaseDatasetTrafo:
    if name in ("mri2d"):
        from src.problem_trafos.dataset_trafo.fastmri_2d_trafo import (
            FastMRI2DDataTransform,
        )

        return FastMRI2DDataTransform(
            device=device,
            provide_measurement=provide_measurement,
            provide_pseudoinverse=provide_pseudoinverse,
            **cfg_kwargs,
        )
    elif name in ("mri3d"):
        from src.problem_trafos.dataset_trafo.fastmri_3d_trafo import (
            FastMRI3DDataTransform,
        )

        return FastMRI3DDataTransform(
            device=device,
            provide_measurement=provide_measurement,
            provide_pseudoinverse=provide_pseudoinverse,
            **cfg_kwargs,
        )
    else:
        raise NotImplementedError(
            "Dataset trafo with name {} not implemented".format(name)
        )
