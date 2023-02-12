import os.path

import numpy as np
import torch
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
from tqdm import tqdm

from homework.modeling.diffusion import DiffusionModel


def train_step(
    model: DiffusionModel, x: torch.Tensor, optimizer: Optimizer, device: str
):
    optimizer.zero_grad()
    x = x.to(device)
    loss = model(x)
    loss.backward()
    optimizer.step()
    return loss


def train_epoch(
    model: DiffusionModel, dataloader: DataLoader, optimizer: Optimizer, device: str
):
    model.train()
    pbar = tqdm(dataloader)
    batch_losses = []
    loss_ema = None
    for x, _ in pbar:
        train_loss = train_step(model, x, optimizer, device)
        batch_losses.append(train_loss.detach().cpu().numpy())
        loss_ema = train_loss if loss_ema is None else 0.9 * loss_ema + 0.1 * train_loss
        pbar.set_description(f"loss: {loss_ema:.4f}")

    return dict(
        mean_loss=np.mean(batch_losses),
        loss_ema=loss_ema,
    )


def generate_samples(model: DiffusionModel, device: str, path: str, epoch_no: int):
    model.eval()
    with torch.no_grad():
        samples = model.sample(8, (3, 32, 32), device=device)
        grid = make_grid(samples, nrow=4)
        if not os.path.exists(path):
            os.makedirs(path)
        save_image(grid, path + f"/{epoch_no:02d}.png")
        return grid
