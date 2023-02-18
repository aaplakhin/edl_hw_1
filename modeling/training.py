import torch
import wandb
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from modeling.diffusion import DiffusionModel


def train_step(model: DiffusionModel, x: torch.Tensor, optimizer: Optimizer, device: str):
    optimizer.zero_grad()
    x = x.to(device)
    loss = model(x)
    loss.backward()
    optimizer.step()
    return loss


def train_epoch(model: DiffusionModel, dataloader: DataLoader, optimizer: Optimizer, device: str):
    model.train()
    pbar = tqdm(dataloader)
    loss_ema = None
    for x, _ in pbar:
        train_loss = train_step(model, x, optimizer, device)
        loss_ema = train_loss if loss_ema is None else 0.9 * loss_ema + 0.1 * train_loss
        pbar.set_description(f"loss: {loss_ema:.4f}")

        wandb.log({'train_loss': train_loss.item(), 'train_loss_ema': train_loss.item()})

    return {'train_loss': train_loss.item(), 'train_loss_ema': train_loss.item()}


def generate_samples(model: DiffusionModel, device: str, path: str, num_epoch: int):
    model.eval()
    with torch.no_grad():
        samples = model.sample(8, (3, 32, 32), device, num_epoch)
        grid = make_grid(samples, nrow=8)
        save_image(grid, path)
        return grid
