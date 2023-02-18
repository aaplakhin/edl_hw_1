import pytest
import torch
import os

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from omegaconf import OmegaConf

from modeling.diffusion import DiffusionModel
from modeling.training import train_step
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
    torch.manual_seed(0)
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
def test_training_epoch_with_hydra(device):
    cfg = OmegaConf.load(f"configs/default.yaml")

    cfg.num_epochs, cfg.device, cfg.cfg_name = 1, device, f"test_1_epoch_{device}"

    if device == "cpu":
        cfg.batch_size = 1
        cfg.test_cpu = True

    losses = main(cfg)

    assert os.path.exists(f"used_configs/used_config_{cfg.cfg_name}.yaml")

    used_cfg = OmegaConf.load(f"used_configs/used_config_{cfg.cfg_name}.yaml")

    assert used_cfg.num_epochs == 1

    if device != "cpu":
        assert losses["train_loss"] < 0.3

        assert losses["train_loss_ema"] < 0.3

        assert os.path.exists("samples/00.png")


def test_with_different_hiden():
    cfg = OmegaConf.load(f"configs/default.yaml")

    cfg.unet_params = [3, 3, 128]
    cfg.num_epochs = 1
    cfg.cfg_name = "test_1_epoch_unet_128"

    losses_original_hs = main(cfg)

    cfg.unet_params = [3, 3, 64]

    cfg.cfg_name = "test_1_epoch_unet_64"

    losses_not_original_hs = main(cfg)

    assert losses_original_hs["train_loss"] < losses_not_original_hs["train_loss"]

    assert losses_original_hs["train_loss_ema"] < losses_not_original_hs["train_loss_ema"]
