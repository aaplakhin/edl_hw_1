import pytest
import torch
import os
import wandb

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from omegaconf import DictConfig, OmegaConf

from modeling.diffusion import DiffusionModel
from modeling.training import train_step, train_epoch
from modeling.unet import UnetModel
from main import main


@pytest.fixture
def train_dataset():
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = CIFAR10(
        "./data",
        train=True,
        download=True,
        transform=tf,
    )
    return dataset


@pytest.mark.parametrize(["device"], [["cpu"], ["cuda"]])
def test_train_on_one_batch(device, train_dataset):
    # note: you should not need to increase the threshold or change the hyperparameters
    ddpm = DiffusionModel(
        eps_model=UnetModel(3, 3, hidden_size=32),
        betas=(1e-4, 0.02),
        num_timesteps=1000,
    )
    ddpm.to(device)

    optim = torch.optim.Adam(ddpm.parameters(), lr=5e-4)
    dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    x, _ = next(iter(dataloader))
    loss = None
    for i in range(50):
        loss = train_step(ddpm, x, optim, device)
    assert loss < 0.5


@pytest.mark.parametrize(["device"], [["cpu"], ["cuda"]])
def test_training_epoch_gpu_with_hydra(device):
    cfg = OmegaConf.load(f"{os.path.dirname(__file__)}/../configs/default.yaml")

    cfg.num_epochs, cfg.device, cfg.cfg_name = 1, device, f"test_1_epoch_{device}"

    if device == "cpu":
        cfg.batch_size = 1
        cfg.test_cpu = True

    losses = main(cfg)

    assert os.path.exists(f"used_configs/used_config_{cfg.cfg_name}.yaml")

    used_cfg = OmegaConf.load(f"{os.path.dirname(__file__)}/../used_configs/used_config_{cfg.cfg_name}.yaml")

    assert used_cfg.num_epochs == 1

    if device != "cpu":
        assert losses["train_loss"] < 0.5

        assert losses["train_loss_ema"] < 0.5

        assert os.path.exists("samples/00.png")
