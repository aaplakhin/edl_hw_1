import os
import torch
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10

from modeling.diffusion import DiffusionModel
from modeling.training import generate_samples, train_epoch
from modeling.unet import UnetModel


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg: DictConfig):
    wandb.login(key=cfg.api_key)
    wandb.init(project=cfg.name, config=OmegaConf.to_container(cfg), name=cfg.cfg_name)

    if not os.path.exists("used_configs/"):
        os.makedirs("used_configs/")

    OmegaConf.save(cfg, f"used_configs/used_config_{cfg.cfg_name}.yaml")

    wandb.run.log_code(root="used_configs", name="used_config",
                       include_fn=lambda path: path.endswith(".yaml"))

    ddpm = DiffusionModel(
        eps_model=UnetModel(*cfg.unet_params),
        betas=cfg.diffusion_params,
        num_timesteps=cfg.num_timesteps,
    )

    ddpm.to(cfg.device)
    if cfg.has_flip:
        train_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(p=cfg.flip_prob),
                transforms.RandomVerticalFlip(p=cfg.flip_prob),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
    else:
        train_transforms = transforms.Compose([transforms.ToTensor(),
                                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    dataset = CIFAR10(
        "cifar10",
        train=True,
        download=True,
        transform=train_transforms,
    )

    wandb.log({"inputs": [wandb.Image(img) for img in dataset.data[:64]]})

    if cfg.test_cpu:
        dataset_cpu = Subset(dataset, [i for i in range(64)])
        dataloader = DataLoader(dataset_cpu, batch_size=cfg.batch_size,
                                num_workers=cfg.num_workers,
                                shuffle=True)
    else:
        dataloader = DataLoader(dataset, batch_size=cfg.batch_size,
                                num_workers=cfg.num_workers,
                                shuffle=True)

    if cfg.optimizer == "adam":
        optim = torch.optim.Adam(ddpm.parameters(), lr=cfg.lr, eps=cfg.eps_adam)
    elif cfg.optimizer == "sgd":
        optim = torch.optim.SGD(ddpm.parameters(), lr=cfg.lr)
    else:
        raise NotImplemented("This project does not support other optimizers")

    if not os.path.exists("samples/"):
        os.makedirs("samples/")

    losses = None
    for i in range(cfg.num_epochs):
        losses = train_epoch(ddpm, dataloader, optim, cfg.device)
        if not cfg.test_cpu:
            generate_samples(ddpm, cfg.device, f"samples/{i:02d}.png", i)

            wandb.log({"samples": wandb.Image(f"samples/{i:02d}.png"),
                       "lr": optim.param_groups[-1]["lr"]})

    wandb.finish()

    return losses


if __name__ == "__main__":
    main()
