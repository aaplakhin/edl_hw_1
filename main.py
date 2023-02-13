import os
import torch
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

from modeling.diffusion import DiffusionModel
from modeling.training import generate_samples, train_epoch
from modeling.unet import UnetModel


def main(cfg: DictConfig):

    wandb.run.log_code(f"conf/{cfg.name}.yaml")

    ddpm = DiffusionModel(
        eps_model=UnetModel(*cfg.unet_params),
        betas=cfg.diffusion_params,
        num_timesteps=cfg.num_timesteps,
    )

    ddpm.to(cfg.device)

    train_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    dataset = CIFAR10(
        "cifar10",
        train=True,
        download=True,
        transform=train_transforms,
    )

    dataloader = DataLoader(dataset, batch_size=cfg.batch_size,
                            num_workers=cfg.num_workers,
                            shuffle=True)

    if cfg.optimizer == "adam":
        optim = torch.optim.Adam(ddpm.parameters(), lr=cfg.lr, eps=cfg.eps_adam)
    elif cfg.optimizer == "sgd":
        optim = torch.optim.SGD(ddpm.parameters(), lr=cfg.lr)
    else:
        raise NotImplemented("This project does not support other optimizers")

    for i in range(cfg.num_epochs):
        train_epoch(ddpm, dataloader, optim, cfg.device)

        if not os.path.exists('samples/'):
            os.makedirs('samples/')

        generate_samples(ddpm, cfg.device, f"samples/{i:02d}.png")

        wandb.log({"Image" : wandb.Image(f"samples/{i:02d}.png")})


def main_cfg(cfg_name: str = "default"):
    @hydra.main(version_base=None, config_path="configs", config_name=cfg_name)
    def _main(cfg: DictConfig):
        main(cfg=cfg)

    _main()


if __name__ == "__main__":
    wandb.init(project='hw_edl')
    main_cfg("default")
