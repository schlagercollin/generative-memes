import torch
from settings import batch_size, workers
from dataset import MemeTemplateDataset
from matplotlib import pyplot as plt
import numpy as np
import torchgan
import torchvision.utils as vutils
import os

from torchgan.models import DCGANGenerator, DCGANDiscriminator

from torchgan.losses import MinimaxDiscriminatorLoss, MinimaxGeneratorLoss, WassersteinGeneratorLoss, WassersteinDiscriminatorLoss, WassersteinGradientPenalty, LeastSquaresGeneratorLoss, LeastSquaresDiscriminatorLoss

from torchgan.trainer import Trainer

from torch import nn
from torch.optim import Adam

os.environ["TENSORBOARD_LOGGING"] = "1"

if __name__ == "__main__":

    dataset = MemeTemplateDataset(epoch_multiplier=1)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True, 
        num_workers=workers
    )

    dcgan_network = {
        "generator": {
            "name": DCGANGenerator,
            "args": {
                "encoding_dims": 100,
                "out_size": 64,
                "out_channels": 3,
                "step_channels": 64,
                "nonlinearity": nn.LeakyReLU(0.2),
                "last_nonlinearity": nn.Tanh(),
            },
            "optimizer": {"name": Adam, "args": {"lr": 0.0001, "betas": (0.5, 0.999)}},
        },
        "discriminator": {
            "name": DCGANDiscriminator,
            "args": {
                "in_size": 64,
                "in_channels": 3,
                "step_channels": 64,
                "nonlinearity": nn.LeakyReLU(0.2),
                "last_nonlinearity": nn.LeakyReLU(0.2),
            },
            "optimizer": {"name": Adam, "args": {"lr": 0.0003, "betas": (0.5, 0.999)}},
        },
    }

    minimax_losses = [MinimaxGeneratorLoss(), MinimaxDiscriminatorLoss()]
    wgangp_losses = [
        WassersteinGeneratorLoss(),
        WassersteinDiscriminatorLoss(),
        WassersteinGradientPenalty(),
    ]
    lsgan_losses = [LeastSquaresGeneratorLoss(), LeastSquaresDiscriminatorLoss()]

    # Plot some of the training images
    # real_batch = next(iter(dataloader))
    # plt.figure(figsize=(8, 8))
    # plt.axis("off")
    # plt.title("Training Images")
    # plt.imshow(
    #     np.transpose(
    #         vutils.make_grid(real_batch, padding=2, normalize=True).cpu(), (1, 2, 0)
    #     )
    # )
    # plt.show()

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        # Use deterministic cudnn algorithms
        torch.backends.cudnn.deterministic = True
        epochs = 10
    else:
        device = torch.device("cpu")
        epochs = 100

    print("Device: {}".format(device))
    print("Epochs: {}".format(epochs))

    trainer = Trainer(
        dcgan_network, lsgan_losses, sample_size=64, epochs=epochs, device=device
    )

    trainer(dataloader)