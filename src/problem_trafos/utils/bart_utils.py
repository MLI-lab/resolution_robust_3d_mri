import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import fastmri
from tqdm import tqdm
import multiprocessing as mp

def import_bart(
    base_path : str
):
    sys.path.insert(0, os.path.join(base_path, "python"))
    os.environ['TOOLBOX_PATH'] = base_path
    import bart

def compute_sens_maps(masked_ksp):
    ### compute sensitivity maps
    masked_ksp = masked_ksp[...,0] + 1j*masked_ksp[...,1]
    sens_maps = bart.bart(1, f'ecalib -d0 -m1', np.array([np.moveaxis(masked_ksp.detach().cpu().numpy(),0,2)]))
    return np.moveaxis(sens_maps[0],2,0)

def compute_sens_maps_3d(masked_ksp):
    kspace_full_complex = torch.view_as_complex(masked_ksp)
    kspace_full_complex_np = kspace_full_complex.moveaxis(0, -1).cpu().numpy() # moved coil dim to last dim
    sens_maps = bart.bart(1, f'ecalib -d0 -m1', kspace_full_complex_np)
    return sens_maps

def compute_sens_maps_np(masked_ksp):
    ### compute sensitivity maps
    masked_ksp = masked_ksp[...,0] + 1j*masked_ksp[...,1]
    sens_maps = bart.bart(1, f'ecalib -d0 -m1', np.array([np.moveaxis(masked_ksp,0,2)]))
    return np.moveaxis(sens_maps[0],2,0)

def compute_sens_maps_mp(masked_ksp, pool_size=8):
    iterates = list(masked_ksp.cpu().numpy()) if torch.is_tensor(masked_ksp) else list(masked_ksp)
    return np.array(mp.Pool(pool_size).map(compute_sens_maps_np, iterates))

def compute_l1_wavelet_solution(kspace, sensmaps, reg_param=4e-4):
    kspace_full_complex = torch.view_as_complex(kspace)
    kspace_full_complex_np = kspace_full_complex.moveaxis(0, -1).cpu().numpy()
    result_np = bart.bart(1, f'pics -l1 -r{reg_param}', kspace_full_complex_np, sensmaps.cpu().squeeze().numpy())
    return torch.view_as_real(torch.from_numpy(result_np)).to(kspace.device)