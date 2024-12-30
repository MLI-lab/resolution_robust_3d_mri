
from src.diffmodels.sampler.base_conditioning_method import ConditioningMethod
from .dds_conditioner import DecomposedDiffusionSampling
from .dps_conditioner import PosteriorSampling

def get_conditioning_method(name : str, **kwargs) -> ConditioningMethod:
    if name == 'dps':
        return PosteriorSampling(**kwargs)
    elif name == 'dds':
        return DecomposedDiffusionSampling(**kwargs)
    else:
        raise NotImplementedError(f'Conditioning method {name} not implemented')