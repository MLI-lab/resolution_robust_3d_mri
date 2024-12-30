from typing import Any, Dict, Optional

import torch

from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb

from src.utils.wandb_utils import tensor_to_wandbimages_dict

from .loss import epsilon_based_loss_fn
from ..sde import SDE
from ..utils_save import save_model

from src.problem_trafos.prior_target_trafo.base_prior_trafo import BasePriorTrafo

from src.diffmodels.archs.std.unet import UNetModel
from src.diffmodels.ema import ExponentialMovingAverage

from src.diffmodels.sampler.base_sampler import BaseSampler
from src.sample_logger.base_sample_logger import BaseSampleLogger
from src.representations.fixed_grid_representation import FixedGridRepresentation

def score_model_trainer(
    score: UNetModel,
    sde: SDE,
    dataloader_train: DataLoader,
    dataloader_val: DataLoader,
    optim_kwargs: Dict,
    val_kwargs: Dict,
    prior_trafo: BasePriorTrafo,
    sampler : BaseSampler,
    sample_logger : BaseSampleLogger,
    device: Optional[Any] = None
    ):
    
    optimizer = Adam(score.parameters(), lr=optim_kwargs['lr'])
    loss_fn = epsilon_based_loss_fn 

    ema = None
    if optim_kwargs.use_ema: 
        ema = ExponentialMovingAverage(
            score.parameters(),
            decay=optim_kwargs['ema_decay']
            )

    batch_size = optim_kwargs['batch_size']

    log_cfg = optim_kwargs["log_dataset_stats_before_training"]
    if log_cfg["enabled"]:
        num_samples = len(dataloader_train) * batch_size if log_cfg["num_dataloader_stat_samples"] < 0 else log_cfg["num_dataloader_stat_samples"]
        num_images = len(dataloader_train) * batch_size if log_cfg["num_dataloader_image_samples"] < 0 else log_cfg["num_dataloader_image_samples"]
        samples_mean = torch.zeros(num_samples)
        samples_std = torch.zeros(num_samples)
        samples_norm = torch.zeros(num_samples)

        with tqdm(enumerate(dataloader_train), total=len(dataloader_train)) as pbar:
            for i, x in pbar:
                if i < num_images:
                    wandb.log({
                        'global_step': i,
                        'step' : i,
                        **tensor_to_wandbimages_dict(f"data_samples_{i}", x, show_phase=False)
                    })

                if i < num_samples:
                    if x.shape[0] != batch_size:
                        continue
                    x = x.view(batch_size,-1)
                    samples_mean[i*batch_size:(i+1)*batch_size] = x.mean(dim=-1)
                    samples_std[i*batch_size:(i+1)*batch_size] = x.std(dim=-1)
                    samples_norm[i*batch_size:(i+1)*batch_size] = torch.linalg.norm(x, dim=-1)

                if i > num_images and i > num_samples:
                    break
        
        wandb.run.summary.update({
            'samples_count': len(dataloader_train),
            'sample_mean_mean': samples_mean.mean(),
            'sample_mean_std': samples_mean.std(),
            'sample_std_mean': samples_std.mean(),
            'sample_std_std': samples_std.std(),
            'sample_norm_mean': samples_norm.mean(),
            'sample_norm_std': samples_norm.std(),
            })

    sample_logger.init_run()

    grad_step = 0
    for epoch in range(optim_kwargs['epochs']):
        avg_loss, num_items = 0, 0
        with tqdm(enumerate(dataloader_train), total=len(dataloader_train)) as pbar:
            score.train()
            for cntr, x in pbar:

                x = x.to(device)

                x = prior_trafo(x)

                loss = loss_fn(
                    x=x,
                    model=score,
                    sde=sde
                    )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pbar.set_description(
                    f'losss={loss.item():.1f}',
                    refresh=False
                )
                
                grad_step += 1
                avg_loss += loss.item() * x.shape[0]
                num_items += x.shape[0]

            should_save_model = (
                epoch % optim_kwargs['save_model_every_n_epoch'] == 0 or epoch == optim_kwargs['epochs'] - 1)
            
            if should_save_model:
                save_model(score=score, epoch=epoch, optim_kwargs=optim_kwargs, ema=ema)
            
            if optim_kwargs.use_ema and (
                grad_step > optim_kwargs['ema_warm_start_steps'] or epoch > 0):
                ema.update(score.parameters())

            def eval_on_validation_set(model):
                with torch.no_grad():
                    model.eval()
                    val_loss = 0
                    val_num_items = 0
                    for x in dataloader_val:
                        x = x.to(device)
                        x = prior_trafo(x)
                        loss = loss_fn(
                            x=x,
                            model=score,
                            sde=sde
                        )
                        val_loss += loss.item() * x.shape[0]
                        val_num_items += x.shape[0]
                    return val_loss / val_num_items

            val_loss = eval_on_validation_set(score)
                        
            wandb.log(
                {'loss': avg_loss / num_items, 'val_loss' : val_loss,  'epoch': epoch + 1, 'step': epoch + 1}
            )

            if val_kwargs.sample_freq is not None:
                if epoch % val_kwargs.sample_freq == 0:
                    if optim_kwargs.use_ema:
                        ema.store(score.parameters())
                        ema.copy_to(score.parameters())
                        score = score.to(device)
                    score.eval()
                    
                    sample_logger.init_sample_log(sample_nr = epoch, mesh = None) # fixed grid ignores mesh

                    sample = sampler.sample()

                    representation = FixedGridRepresentation(in_shape=tuple(sample.shape[:-1]), out_features=sample.shape[-1], warm_start=sample)
                    sample_logger.close_sample_log(representation=representation)

                    val_loss_ema = eval_on_validation_set(score)

                    if wandb.run is not None:
                        wandb.run.log({
                            'sample_mean': sample.mean(),
                            'sample_std': sample.std(),
                            "val_loss_ema": val_loss_ema
                            })

                    if optim_kwargs.use_ema: ema.restore(score.parameters())

    torch.save(score.state_dict(), 'last_model.pt')

    sample_logger.close_run()
