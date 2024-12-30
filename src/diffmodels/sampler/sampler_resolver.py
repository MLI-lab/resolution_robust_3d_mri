
from .ddpm_sampler import DDPM
from .ddim_sampler import DDIM
from .base_sampler import BaseSampler

def get_sampler(name : str, **cfg_kwargs) -> BaseSampler:
    if name == "ddpm":
        return DDPM(**cfg_kwargs)
    elif name == "ddim":
        return DDIM(**cfg_kwargs)
    else:
        raise NotImplementedError(f"Sampler {name} not implemented")