from typing import Optional, Any, Tuple
from torch import Tensor

from src.representations.mesh import SliceableMesh
from src.representations.gaussian.gaussian_model import GaussianModel
from src.representations.base_coord_based_representation import CoordBasedRepresentation

class GaussianRepresentation(CoordBasedRepresentation):
    def __init__(self, 
            model_params,
            opt_params,
            warm_start: Optional[Tensor] = None,
            warm_start_mesh : Optional[Tensor] = None,
            warm_start_cfg : Optional[Tensor] = None,
            device : Optional[Any] = None):
        super().__init__()

        self.device = device
        self.gaussian_model = GaussianModel(device=device, model_params=model_params, opt_params= opt_params)

        warm_start_iters = warm_start_cfg.warmstart_iters

        if warm_start is not None and warm_start_mesh is not None and warm_start_iters is not None:
            self.gaussian_model.initialize_with_image(mesh=warm_start_mesh, warmstart_voxel_rep=warm_start, warmstart_iters=warm_start_iters)
        else:
            raise NotImplementedError("Initialization without warm_start currently not implemented.")

    def scale_learning_rates(self, factor):
        self.gaussian_model.scale_learning_rates(factor)

    def forward(self, mesh : SliceableMesh) -> Tensor:
        volume = self.gaussian_model.rasterize(mesh)
        return mesh.apply_slicing_to_tensor(volume)

    def forward_splitted(self, mesh : SliceableMesh, custom_device : Optional[Any] = None, split : int = 1) -> Tensor:
        return self.forward(mesh)

    def get_optimizer_params(self) -> Tuple:
        return self.gaussian_model.optimizer_params