# Source: https://github.com/phlippe/uvadlc_notebooks/blob/master/docs/tutorial_notebooks/tutorial9/AE_CIFAR10.ipynb

from nets import Encoder, Decoder
import torch.nn as nn
import torch.nn.functional as F
import torch
import lightning as L
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


class Autoencoder(L.LightningModule):
    def __init__(
        self,
        latent_dim: int,
        width: int = 224,
        height: int = 224,
    ):
        super().__init__()
        # Saving hyperparameters of autoencoder
        self.save_hyperparameters()
        # Creating encoder and decoder
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        # Example input array needed for visualizing the graph of the network
        self.example_input_array = torch.zeros(2, 3, width, height)

    def forward(self, x):
        """The forward function takes in an image and returns the reconstructed image."""
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def _get_reconstruction_loss(self, batch):
        """Given a batch of images, this function returns the reconstruction loss (MSE in our case)"""
        x, _ = batch  # We do not need the labels
        x_hat = self.forward(x)
        loss = F.mse_loss(x, x_hat, reduction="none")
        loss = loss.sum(dim=[1, 2, 3]).mean(dim=[0])
        return loss

    def calc_metrics(self, batch):
        """Given a batch of images, this functions returns the PSNR, SSIM and LPIPS"""
        x, _ = batch
        x_hat = self.forward(x)

        psnr = PeakSignalNoiseRatio()
        ssim = StructuralSimilarityIndexMeasure()
        try:
            lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg", reduction='mean', normalize=True)
            lpips_val = lpips(x_hat, x)
        except Exception as e:
            self.log("Error while calculating LPIPS")
            self.log("Details", str(e))
            lpips_val = 0

        return psnr(x_hat, x), ssim(x_hat, x), lpips_val

    def configure_optimizers(self):
        optimizer = nn.optim.Adam(self.parameters(), lr=1e-3)
        # Using a scheduler is optional but can be helpful.
        # The scheduler reduces the LR if the validation performance hasn't improved for the last N epochs
        scheduler = nn.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.2, patience=20, min_lr=5e-5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def training_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log("test_loss", loss)
        psnr, ssim, lpips = self.calc_metrics(batch)
        self.log("psnr", psnr)
        self.log("ssim", ssim)
        self.log("lpips", lpips)
