from typing import Optional, Dict, List, Any
from functools import partial
from abc import ABC, abstractmethod

import torch 
from torch import Tensor

from src.reconstruction.variational.noise_loss import noise_loss

from src.diffmodels.archs.std.unet import UNetModel
from src.diffmodels import SDE
from src.problem_trafos.fwd_trafo.base_fwd_trafo import BaseFwdTrafo
from src.problem_trafos.prior_target_trafo.base_prior_trafo import BasePriorTrafo

from src.representations.slice_methods import SliceMethod
from src.representations.base_coord_based_representation import CoordBasedRepresentation
from src.representations.mesh import SliceableMesh

import pywt, ptwt

class BaseVariationalObjective(ABC):
    def __init__(
        self,
        observation: Tensor, 
        mesh_data_con: SliceableMesh,
        mesh_data_reg: Optional[SliceableMesh],
        fwd_trafo: BaseFwdTrafo,
        prior_trafo : BasePriorTrafo,
        steps_data_con : List[int],
        steps_data_reg : List[int],
        slice_method_data_con: Optional[SliceMethod],
        slice_method_prior_reg: Optional[SliceMethod],
        outer_iterations_max : Optional[int] = None,
        score : Optional[UNetModel] = None,
        sde : Optional[SDE] = None
    ):
        self.observation = observation
        self.mesh_data_con = mesh_data_con
        self.mesh_data_reg = mesh_data_reg
        self.fwd_trafo = fwd_trafo
        self.prior_trafo = prior_trafo
        self.steps_data_con = steps_data_con
        self.steps_data_reg = steps_data_reg
        self.slice_method_data_con = slice_method_data_con
        self.slice_method_prior_reg = slice_method_prior_reg
        self.outer_iterations_max = outer_iterations_max
        self.score = score
        self.sde = sde

    @abstractmethod
    def __call__(
        self,
        representation: CoordBasedRepresentation,
        outer_iteration : int,
        inner_iteration : int,
        ) -> Tensor:
        pass

class NoPriorObjective(BaseVariationalObjective):
    def __init__(
        self,
        **base_obj_kwargs
    ):
        super().__init__(**base_obj_kwargs)

    def __call__(
        self,
        representation: CoordBasedRepresentation,
        outer_iteration : int,
        inner_iteration : int,
        ) -> Tensor:

        criterion = torch.nn.MSELoss()

        if inner_iteration in self.steps_data_con:
            if self.slice_method_data_con is not None:

                slices, slice_inds = self.slice_method_data_con(representation, self.mesh_data_con, self.mesh_data_con, outer_iteration=outer_iteration, inner_iteration=inner_iteration)
                assert len(slices) == 1, "Currently only slices from one direction are supported"
                slice_volume_index = self.slice_method_data_con.volume_indices[0]
                slice_volume_index_rev = slice_volume_index - slices[0].ndim 

                datafit = criterion(
                    self.fwd_trafo.trafo(slices[0], slice_inds[0].int(), slice_volume_index),
                    self.observation.index_select(slice_volume_index_rev, slice_inds[0].int()) # add one to index since coil dim is added, or not one if it is not in kspace
                ) / len(self.steps_data_con)

            else:
                datafit = criterion(
                    self.fwd_trafo(representation.forward(self.mesh_data_con)), 
                    self.observation
                ) / len(self.steps_data_con)
        else:
            datafit = torch.zeros(1, device=representation.device)

        return datafit, datafit.item(), 0.0

class L1WaveletObjective(BaseVariationalObjective):
    def __init__(
        self,
        name: Optional[str],
        wavelet_type : str,
        wavelet_level : int,
        wavelet_axis : List[int],
        norm_order : int,
        reg_strength : float,
        **base_obj_kwargs
    ):
        super().__init__(**base_obj_kwargs)
        self.name = name
        self.wavelet_type = wavelet_type
        self.wavelet_level = wavelet_level
        self.wavelet_axis = wavelet_axis
        self.norm_order = norm_order
        self.reg_strength = reg_strength

    def __call__(
        self,
        coord_rep: CoordBasedRepresentation,
        outer_iteration : int,
        inner_iteration : int,
        ) -> Tensor:

        criterion = torch.nn.MSELoss()

        if inner_iteration in self.steps_data_con:
            datafit = criterion(
                self.fwd_trafo(coord_rep.forward(self.mesh_data_con)), 
                self.observation
            ) / len(self.steps_data_con)
        else:
            datafit = torch.zeros(1, device=coord_rep.device)

        if inner_iteration in self.steps_data_reg:
            wt_coeffs = ptwt.wavedec3(
                self.prior_trafo(coord_rep.forward(self.mesh_data_reg)),
                pywt.Wavelet(self.wavelet_type),level=self.wavelet_level, axes=self.wavelet_axis)[0]
            regfit = self.reg_strength * torch.norm(wt_coeffs, p=self.norm_order) / len(self.steps_data_reg)
        else:
            regfit = torch.zeros(1, device=coord_rep.device)

        return datafit + regfit, datafit.item(), regfit.item()

class DiffusionVariationanlObjective(BaseVariationalObjective):
    def __init__(
        self,
        name : str,
        reg_strength : float,
        adapt_reg_strength : bool,
        steps_scaler : float,
        time_sampling_method : str,
        score : UNetModel,
        sde : SDE,
        **base_obj_kwargs
    ):
        super().__init__(**base_obj_kwargs)
        self.name = name
        self.reg_strength = reg_strength
        self.adapt_reg_strength = adapt_reg_strength
        self.steps_scaler = steps_scaler
        self.time_sampling_method = time_sampling_method
        self.score = score
        self.sde = sde

    def __call__(
        self,
        coord_rep: CoordBasedRepresentation, 
        outer_iteration : int,
        inner_iteration : int,
        ) -> callable:

        criterion = torch.nn.MSELoss()

        if inner_iteration in self.steps_data_con:
            if self.slice_method_data_con is not None:
                slices, slice_inds = self.slice_method_data_con(coord_rep, self.mesh_data_con, self.mesh_data_con, outer_iteration=outer_iteration, inner_iteration=inner_iteration)

                assert len(slices) == 1, "Currently only one slice is supported"
                slice_volume_index = self.slice_method_data_con.volume_indices[0]
                slice_volume_index_rev = slice_volume_index - slices[0].ndim

                datafit = criterion(
                    self.fwd_trafo.trafo(slices[0], slice_inds[0].int(), slice_volume_index),
                    self.observation.index_select(slice_volume_index_rev, slice_inds[0].int())
                ) / len(self.steps_data_con)

            else:
                datafit = criterion(
                    self.fwd_trafo(coord_rep.forward(self.mesh_data_con)), 
                    self.observation
                ) / len(self.steps_data_con)
        else:
            datafit = torch.zeros(1, device=coord_rep.device)

        if inner_iteration in self.steps_data_reg:
            nl = partial(noise_loss,
                outer_iteration=outer_iteration,
                outer_iterations_max=self.outer_iterations_max,
                score=self.score,
                sde=self.sde, 
                repetition=1,
                reg_strength=self.reg_strength,
                adapt_reg_strength=self.adapt_reg_strength,
                steps_scaler=self.steps_scaler,
                time_sampling_method=self.time_sampling_method
                )

            if self.slice_method_prior_reg is not None:
                slices, slice_inds = self.slice_method_prior_reg(coord_rep, self.mesh_data_reg, self.mesh_data_con, outer_iteration=outer_iteration, inner_iteration=inner_iteration)
            else:
                slices = [coord_rep.forward(self.mesh_data_reg)] 

            regfit = sum([nl(self.prior_trafo(slice)).mean() for slice in slices]) / len(slices) / len(self.steps_data_reg)
        else:
            regfit = torch.zeros(1, device=coord_rep.device)

        return datafit + regfit, datafit.item(), regfit.item()



def get_variational_objective(
    # ray_trafo: BaseFwdTrafo,
    # observation: Tensor,
    # steps_data_con : List[int],
    # steps_data_reg : List[int],
    # mesh_data_con : SliceableMesh,
    # mesh_data_reg : SliceableMesh,
    # prior_trafo : Optional[BasePriorTrafo] = None,
    # score: Optional[UNetModel] = None, 
    # sde: Optional[DDPM] = None, 
    # slice_method_data_con: Optional[SliceMethod] = None,
    # slice_method_prior_reg: Optional[SliceMethod] = None,
    # outer_iterations_max : Optional[int] = None,
    cfg_regularization: Optional[Dict[str, Any]] = None,
    **base_reg_kwargs
    ) ->  BaseVariationalObjective:

    if cfg_regularization.name is None:
        return NoPriorObjective(
            **base_reg_kwargs
        )
        # return partial(
            # _no_prior_criterion,
            # observation=observation,
            # mesh_data_con=mesh_data_con,
            # fwd_trafo=ray_trafo,
            # steps_data_con=steps_data_con,
            # slice_method_data_con=slice_method_data_con
        # )
    elif cfg_regularization.name == 'diffusion':
        return DiffusionVariationanlObjective(
            **cfg_regularization,
            **base_reg_kwargs
        )
        # return partial(
            # _diffusion_reg_criterion, 
            # outer_iterations_max=outer_iterations_max,
            # observation=observation, 
            # steps_data_con=steps_data_con,
            # steps_data_reg=steps_data_reg,
            # mesh_data_con=mesh_data_con,
            # mesh_data_reg=mesh_data_reg,
            # fwd_trafo=ray_trafo,
            # prior_trafo=prior_trafo,
            # score=score, 
            # sde=sde,
            # slice_method_data_con=slice_method_data_con,
            # slice_method_prior_reg=slice_method_prior_reg,
            # **cfg_regularization
            # )
    elif cfg_regularization.name == "lpwavelet":
        return L1WaveletObjective(
            **cfg_regularization,
            **base_reg_kwargs
        )
        # return partial(
            # _l1wavelet_criterion, 
            # observation=observation, 
            # steps_data_con=steps_data_con,
            # steps_data_reg=steps_data_reg,
            # mesh_data_con=mesh_data_con,
            # mesh_data_reg=mesh_data_reg,
            # fwd_trafo=ray_trafo,
            # prior_trafo=prior_trafo,
            # **cfg_regularization
        # )