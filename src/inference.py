# Source: https://github.com/phlippe/uvadlc_notebooks/blob/master/docs/tutorial_notebooks/tutorial9/AE_CIFAR10.ipynb

from torchvision import transforms
from torchvision.datasets import CIFAR10
import lightning as L
import torch
import torch.utils.data as data
import os
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from utils import GenerateCallback, get_train_images, visualize_reconstructions
from model import Autoencoder


# Transformations applied on each image => only make them a tensor
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Loading the training dataset. We need to split it into a training and validation part
train_dataset = CIFAR10(root=DATASET_PATH, train=True,
                        transform=transform, download=True)
L.seed_everything(42)
train_set, val_set = torch.utils.data.random_split(
    train_dataset, [45000, 5000])

# Loading the test set
test_set = CIFAR10(root=DATASET_PATH, train=False,
                   transform=transform, download=True)

# We define a set of data loaders that we can use for various purposes later.
train_loader = data.DataLoader(
    train_set, batch_size=256, shuffle=True, drop_last=True, pin_memory=True, num_workers=4)
val_loader = data.DataLoader(
    val_set, batch_size=256, shuffle=False, drop_last=False, num_workers=4)
test_loader = data.DataLoader(
    test_set, batch_size=256, shuffle=False, drop_last=False, num_workers=4)

# Create a PyTorch Lightning trainer with the generation callback
trainer = L.Trainer(
    default_root_dir=os.path.join(CHECKPOINT_PATH, "cifar10_%i" % latent_dim),
    accelerator="auto",
    devices=1,
    max_epochs=500,
    callbacks=[
        ModelCheckpoint(save_weights_only=True),
        GenerateCallback(get_train_images(8), every_n_epochs=10),
        LearningRateMonitor("epoch"),
    ],
)
# If True, we plot the computation graph in tensorboard
trainer.logger._log_graph = True
# Optional logging argument that we don't need
trainer.logger._default_hp_metric = None

# Check whether pretrained model exists. If yes, load it and skip training
pretrained_filename = os.path.join(
    CHECKPOINT_PATH, "cifar10_%i.ckpt" % latent_dim)
if os.path.isfile(pretrained_filename):
    print("Found pretrained model, loading...")
    model = Autoencoder.load_from_checkpoint(pretrained_filename)
else:
    model = Autoencoder(base_channel_size=32, latent_dim=latent_dim)
    trainer.fit(model, train_loader, val_loader)
# Test best model on validation and test set
val_result = trainer.test(model, dataloaders=val_loader, verbose=False)
test_result = trainer.test(model, dataloaders=test_loader, verbose=False)
result = {"test": test_result, "val": val_result}

input_imgs = get_train_images(4)
visualize_reconstructions(model, input_imgs)
