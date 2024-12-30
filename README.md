# Resolution-Robust 3D MRI Reconstruction with 2D Diffusion Priors: Diverse-Resolution Training Outperforms Interpolation

This repository contains implementations and code for our paper [**Resolution-Robust 3D MRI Reconstruction with 2D Diffusion Priors: Diverse-Resolution Training Outperforms Interpolation**](https://arxiv.org/abs/2412.18584), available at arXiv.


## Running the code

The code can be run via the following steps. First, three manual steps are required for setting up and configuration. Then, the models can be trained and reconstruction methods performed with the pretrained models.

1. [Setup](#setup): Installing the conda environment, the BART toolbox, external packages and compiling the local CUDA kernels.
3. [Dataset](#datasets): Downloading the Stanford, Calgary-Campinas and/or AHEAD datasets.
2. [Configuration](#config): Configuring where datasets are stored, artifacts shall be stored or loaded from, and experiment tracking.
4. [Model training](#config):  Pretraining the diffusion models on 2D slices on different resolutions.
5. [3D reconstruction](#reconstruction) Reconstruction with the different methods and baselines discussed in the paper.

The repository is structured as follows:
```
 ├───hydra                      # configuration (modified in step 3, used in steps 4,5)
 ├───setup                      # setup (env, setup script) (step 1)
 ├───src                        # python code (used in steps 4,5)
 ├───submodules                 # CUDA code for Gaussian interpolation (compiled in step 1)
 ├───reconstruct.py             # entry point for reconstruction (step 5)
 ├───train_diff_models.py       # entry point for training the models (step 4)
```

<a name="setup"></a>
### 1. Setup
The code has been tested using the `pytorch/pytorch` docker image, with the docker service running on `Linux: Ubuntu 20.04 LTS`. The docker image contains a conda installation, and you can install the environment used for this work using the following script:
```bash
bash setup/setup.sh
conda activate res_rob
```
In addition to setting up the environment (`res_rob`) using the specification `setup/env.yml`, the script installs compilers, the BART toolbox, and compiles the CUDA kernel for our adaptation of Gaussian splatting.

<a name="datasets"></a>
### 2. Datasets
In our paper, we consider the following three datasets:
 - [Stanford (3T) 3D knee volumes](http://mridata.org/)
 - [Calgary-Campinas (3T) 3D brain volumes ](https://portal.conp.ca/dataset?id=projects/calgary-campinas)
 - [AHEAD ultra-high field (7T) 3D brain volumes](https://dataverse.nl/dataset.xhtml?persistentId=doi:10.34894/IHZGQM)

The path to the MRI acquisitions needs to be set in the configuration. Further data preprocessing, such as resolution-changes, is handled in the dataset transformation part of the training and reconstruction pipelines.

<a name="config"></a>
### 3. Configuration

Most aspects of the software can be configured using [Hydra](https://github.com/facebookresearch/hydra), including:
- **Wandb logging** (Optional): If you want to have training and reconstruction runs logged to wandb, set the `entity` and `project` keys in the `hydra/config.yaml` file, and set `log: True`. Wandb will ask for an API key on first run.
- **Hydra outputs** (Optional): For each run (training or reconstruction) Hydra is configured to generate a separate directory. You can configure the base directory for single runs or sweeps in `hydra/cluster/default.yaml`.
- **Dataset paths**: Before starting training, store the paths to the dataset splits in the `hydra/datasets` files, respectively. For each path (e.g. train split of the dataset), state key-value pairs, with the key being the cluster name, and the value the path, respectively.

<a name="training"></a>
### 4. Model training

Diffusion models for reconstruction are trained using the `train_diff_models.py` file. In each of the `hydra/exps` folder there is a `train_dense` or `train_depthwise` configuration file, which can be passed to `train_diff_models.py` to train a model. Alternatively, hydra's sweep capabilities can be used to train multiple models in parallel, using one of the sweeps defined in `hydra/sweeps`.

<a name="reconstruction"></a>
### 5. Reconstruction

Reconstruction can be performed using the `reconstruct.py` file. Similar to training, there is a `varrecon_voxelrep_diffprior` file, for standard reconstruction and variants for the other methods used in the paper. See the configurations in the `sweep` directory for performing performing reconstruction at several accelerations for example.

Note, that before reconstruction you need to set the path to the ema-model in the `base_recon` file, of the respective `exps`.

<a name="citations"></a>
##  Citation
If you use this repository, please cite the paper:
```bibtex
@article{citekey,
    author = {Anselm Krainovic and Stefan Ruschke and Reinhard Heckel},
    title = {{Resolution-Robust 3D MRI Reconstruction with 2D Diffusion Priors: Diverse-Resolution Training Outperforms Interpolation}},
    journal = {arxiv:2412.18584},
    year = {2024}
}
```

<a name="license"></a>
##  License
This project is covered by **Apache 2.0 License**.

