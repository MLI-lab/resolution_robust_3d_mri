from typing import Any, Optional
import torch
import numpy as np
import tqdm
import torch
from simple_knn._C import distCUDA2
from gaussian_rasterizer_complex import ComplexGaussianRasterizer, ComplexGaussianRasterizationSettings
import logging
from src.reconstruction.utils.metrics import PSNR
from functools import partial

def scaled_sigmoid(x, min, max):
    return min + (max - min) * torch.sigmoid(x)

def inverse_scaled_sigmoid(y, min, max):
    return torch.log((y - min) / (max - y))

class GaussianModel:

    def setup_functions(self):

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        if self.use_phase_add_as_imag:
            iden = lambda x:x
            self.opacity_activation = iden
            self.inverse_opacity_activation = iden

            self.phase_add_activation = iden
            self.inverse_phase_add_activation = iden
        else:
            self.opacity_activation = torch.exp
            self.inverse_opacity_activation = torch.log

            self.phase_add_activation = partial(scaled_sigmoid, min=-torch.pi, max=torch.pi)
            self.inverse_phase_add_activation = partial(inverse_scaled_sigmoid, min=-torch.pi, max=torch.pi)

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, device : Any, model_params, opt_params):
        
        self._xyz = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._is_complex = model_params.is_complex
        assert model_params.is_complex, "Only complex representations are supported."
        self._phase = torch.empty(0)
        self._phase_add = torch.empty(0)
        self.use_phase_add_as_imag = model_params.use_phase_add_as_imag

        self.num_out_features = model_params.num_out_features
        self.num_in_features = model_params.num_in_features

        assert self.num_in_features == 3, "The rasterizer currently only supports 3D representations."

        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.device = device

        self.setup_functions()
        self.opt_params = opt_params
        self.model_params = model_params

    def _get_rasterizer(self, mesh_lb, mesh_ub, mesh_resolutions, voxel_offsets, warmup = False):
        self.raster_settings = ComplexGaussianRasterizationSettings(
            scale_multiplier=self.opt_params.scaling_multiplier if not warmup else self.opt_params.scaling_multiplier_warmup,
            use_phase_add_as_imag=self.model_params.use_phase_add_as_imag,
            mesh_lb=mesh_lb,
            mesh_ub=mesh_ub,
            mesh_resolutions=mesh_resolutions,
            voxel_offset_factors=voxel_offsets,
            grad_padding_factor=1.0 / mesh_resolutions.prod().item(),
            grad_padding_const=0.0
        )
        return ComplexGaussianRasterizer(self.raster_settings)

    def rasterize(self, mesh) -> torch.Tensor:
        return self._rasterize(
            mesh_lb=mesh.lower_coords.to(self.device),
            mesh_ub=mesh.upper_coords.to(self.device),
            mesh_resolutions=torch.tensor(
                mesh.matrix_size, device=self.device
            ).float(),
            voxel_offsets=torch.zeros_like(mesh.lower_coords).to(self.device),
        )

    def _rasterize(self, mesh_lb : Optional[torch.Tensor] = None,
            mesh_ub : Optional[torch.Tensor] = None,
            mesh_resolutions : Optional[torch.Tensor] = None,
            voxel_offsets : Optional[torch.Tensor] = None,
            warmup : bool = False
        ) -> torch.Tensor:

        rasterizer = self._get_rasterizer(mesh_lb, mesh_ub, mesh_resolutions, voxel_offsets, warmup=warmup)
    
        gm_xyz = self.get_xyz.float()
        gm_opacities = self.get_opacity.float()
        gm_scaling = self.get_scaling.float()
        gm_rotation = self.get_rotation.float()
        gm_phases = self.get_phase.float()
        gm_phases_add = self.get_phase_add.float()
        return rasterizer(gm_xyz, gm_opacities, gm_scaling, gm_rotation, gm_phases, gm_phases_add)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_phase(self): 
        return self._phase

    @property
    def get_phase_add(self): 
        return self.phase_add_activation(self._phase_add)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def initialize_with_image(self, mesh : torch.Tensor, warmstart_voxel_rep : torch.Tensor, warmstart_iters : int):
        coord_mesh = mesh.get_coord_mesh()
        mesh_reshaped = coord_mesh.cpu().to(self.device).reshape(-1, 3)
        warmstart_reshaped = warmstart_voxel_rep.reshape(-1, self.num_out_features)

        if self.opt_params.random_subset:

            subset = int(self.opt_params.subset_size)
            
            subset_mesh_selection = int(min(subset, np.prod(mesh.matrix_size)))
            rnd_index = torch.randint(0, mesh_reshaped.shape[0], (subset_mesh_selection,), device=self.device)
            mesh_selected = mesh_reshaped.index_select(0, rnd_index)
            warmstart_selected = warmstart_reshaped.index_select(0, rnd_index)

            subset_diff = subset - subset_mesh_selection
            if subset_diff > 0:
                print(f"Random subset size of {subset_mesh_selection} is smaller than requested subset size of {subset}.")

                mesh_added = 2.0 *  torch.rand((subset_diff, 3), device=self.device) - 1.0
                warmstart_added = warmstart_voxel_rep.max() * torch.rand((subset_diff, self.num_out_features), device=self.device)

                mesh_selected = torch.cat((mesh_selected, mesh_added), dim=0)
                warmstart_selected = torch.cat((warmstart_selected, warmstart_added), dim=0)

        else:
            mesh_selected = mesh_reshaped
            warmstart_selected = warmstart_reshaped

        self._create_from_pcd(point_cloud_tensor=mesh_selected, init_data=warmstart_selected)

        self._training_setup(warmup_phase=True)

        if warmstart_iters > 0:
            logging.info(f"Starting warmstart phase with {warmstart_iters} iterations.")

        optimizer = torch.optim.Adam(self.optimizer_params, lr=0.0, eps=1e-15)
        psnr_threshold = self.opt_params.warmup_psnr_threshold
        with tqdm.tqdm(range(warmstart_iters), desc='warmstart') as pbar:
            for i in pbar:
                optimizer.zero_grad()

                results = self.rasterize(mesh)

                loss = (warmstart_voxel_rep - results).square().mean()
                loss.backward()

                optimizer.step()

                psnr = PSNR(results.detach(), warmstart_voxel_rep.detach())

                pbar.set_description(f'psnr={psnr:.1f}', refresh=False)

                if psnr_threshold is not None:
                    if psnr > psnr_threshold:
                        logging.info(f"Warmstart phase finished early due to PSNR threshold ({psnr} > {psnr_threshold}).")
                        break

        self._training_setup(warmup_phase=False)

    def initialize_randomly(self):
        raise NotImplementedError("Not implemented yet")

    def _create_from_pcd(self, point_cloud_tensor, init_data : Optional[torch.Tensor] = None):
        fused_point_cloud = point_cloud_tensor.to(self.device)
        print("Number of points at initialisation : ", fused_point_cloud.shape[0])
        # for some reason distCUDA2 only works on GPU0
        # for small number of points distCUDA2 doesn't provide meaningful results
        if fused_point_cloud.shape[0] > 5:
            dist2 = torch.clamp_min(distCUDA2(point_cloud_tensor.cpu().cuda()), 0.0000001).cpu().to(self.device)
            scales = self.scaling_inverse_activation(self.opt_params.scaling_multiplier_init * torch.sqrt(dist2))[...,None].repeat(1, 3)
        else:
            # bit random
            scales = self.opt_params.scaling_multiplier_init * torch.ones((fused_point_cloud.shape[0], 3), dtype=torch.float, device=self.device)

        rots = torch.zeros((fused_point_cloud.shape[0], 4), device=self.device)
        rots[:, 0] = 1

        if self._is_complex:
            if self.use_phase_add_as_imag:
                opac = self.inverse_opacity_activation(init_data[:,0])
                phases_add = self.inverse_phase_add_activation(init_data[:,1])
            else:
                opac = self.inverse_opacity_activation(init_data.norm(dim=-1))
                phases_add = self.inverse_phase_add_activation(torch.view_as_complex(init_data).angle().unsqueeze(dim=-1))

            phases = torch.zeros_like(fused_point_cloud)
        else:
            opac = self.inverse_opacity_activation(init_data)

        self._xyz = torch.nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._scaling = torch.nn.Parameter(scales.requires_grad_(True))
        self._rotation = torch.nn.Parameter(rots.requires_grad_(True))
        self._opacity = torch.nn.Parameter(opac.requires_grad_(True))
        if self._is_complex:
            self._phase_add = torch.nn.Parameter(phases_add.requires_grad_(True))
            self._phase = torch.nn.Parameter(phases.requires_grad_(True))

    def _training_setup(self, warmup_phase : bool = False):
        training_args = self.opt_params
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=self.device)
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=self.device)

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr if not warmup_phase else training_args.position_lr_warmup, "name": "xyz"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr if not warmup_phase else training_args.opacity_lr_warmup, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr if not warmup_phase else training_args.scaling_lr_warmup, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr if not warmup_phase else training_args.rotation_lr_warmup, "name": "rotation"}
        ]
        if self._is_complex:
            l.append({'params': [self._phase_add], 'lr': training_args.phase_add_lr if not warmup_phase else training_args.phase_add_lr_warmup, "name": "phase_add"})
            if not self.use_phase_add_as_imag:
                l.append({'params': [self._phase], 'lr': training_args.phase_lr if not warmup_phase else training_args.phase_lr_warmup, "name": "phase"})

        self.optimizer_params = l