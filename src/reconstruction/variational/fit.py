# %%
from typing import Dict, Any

import torch
from torch.optim import lr_scheduler, Optimizer
from tqdm import tqdm

from src.representations.base_coord_based_representation import CoordBasedRepresentation
from src.reconstruction.variational.var_objectives import BaseVariationalObjective
from src.sample_logger.base_sample_logger import BaseSampleLogger

def resolve_lr_scheduler(optimizer : Optimizer, name : str, **cfg_kwargs):
    if name == "ReduceLROnPlateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, **cfg_kwargs)
    elif name == "MultiStepLR":
        scheduler = lr_scheduler.MultiStepLR(optimizer, **cfg_kwargs)
    elif name == "CosineAnnealingLR":
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, **cfg_kwargs)
    elif name == "CosineAnnealingWarmRestarts":
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **cfg_kwargs)
    else:
        raise NotImplementedError(
            f'lr_scheduler {["lr_scheduler"]} not implemented'
        )
    return scheduler


def fit(
    representation: CoordBasedRepresentation,
    var_objective: BaseVariationalObjective,
    sample_logger: BaseSampleLogger,
    cfg_fitting: Dict["str", Any],
) -> CoordBasedRepresentation:

    optim_kwargs = cfg_fitting["optimizer"]
    scheduler_kwargs = cfg_fitting["lr_scheduler"]

    # representation.train()
    optimizer = torch.optim.Adam(
        representation.get_optimizer_params(),
        lr=optim_kwargs["lr"],
        betas=(0.9, 0.99),
        weight_decay=0.0,
    )

    if scheduler_kwargs is not None:
        scheduler = resolve_lr_scheduler(optimizer, **scheduler_kwargs)
    else:
        scheduler = None

    max_steps_data_reg = (
        max(optim_kwargs["gradient_acc_steps_prior_reg"])
        if len(optim_kwargs["gradient_acc_steps_prior_reg"]) > 0
        else 0
    )
    max_iteration = max(
        max(optim_kwargs["gradient_acc_steps_data_con"]), max_steps_data_reg
    )

    with tqdm(range(optim_kwargs["iterations"]), desc="coord-net") as pbar:

        for i in pbar:

            optimizer.zero_grad()

            # gradient accumulation
            datafit_it = torch.zeros(1, device=representation.device)
            regfit_it = torch.zeros(1, device=representation.device)
            for j in range(0, max_iteration + 1):
                loss, datafit, regfit = var_objective(representation, i, j)
                datafit_it += datafit
                regfit_it += regfit
                loss.backward()
            loss_it = datafit_it + regfit_it

            if optim_kwargs["clip_grad_max_norm"] is not None:
                torch.nn.utils.clip_grad_norm_(
                    representation.parameters(),
                    max_norm=optim_kwargs["clip_grad_max_norm"],
                )

            if scheduler is not None:
                if scheduler_kwargs.name == "ReduceLROnPlateau":
                    scheduler.step(loss_it.item())
                else:
                    scheduler.step()

            if sample_logger is not None:
                sample_logger(
                    representation=representation,
                    step=i,
                    pbar=pbar,
                    log_dict={
                        "loss": loss_it.item(),
                        "datafit": datafit_it,
                        "regfit": regfit_it,
                        "lr": optimizer.param_groups[0]["lr"],
                    },
                )

            optimizer.step()

    return representation
