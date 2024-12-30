from typing import Optional, Dict

import torch


from src.diffmodels.archs.std.unet import UNetModel
from src.diffmodels.ema import ExponentialMovingAverage

def save_model(score: UNetModel, epoch: int, optim_kwargs: Dict, ema: Optional[ExponentialMovingAverage] = None) -> None:

        model_filename = 'model.pt' if epoch == optim_kwargs['epochs'] - 1 else f'model_{epoch}.pt'
        torch.save(score.state_dict(), model_filename)
        if ema is not None:
            ema_filename = 'ema_model.pt' if epoch == optim_kwargs['epochs'] - 1 else f'ema_model_{epoch}.pt'
            torch.save(ema.state_dict(), ema_filename)
