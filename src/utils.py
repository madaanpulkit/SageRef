# Source: https://github.com/phlippe/uvadlc_notebooks/blob/master/docs/tutorial_notebooks/tutorial9/AE_CIFAR10.ipynb

import torch
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from lightning.pytorch.callbacks import Callback
import torchvision.transforms as transforms
import os
import random
import csv
from PIL import Image
from tqdm.auto import tqdm


class InvalidSplitsError(Exception):
    pass


class GenerateCallback(Callback):
    def __init__(self, imgs, every_n_epochs=1):
        super().__init__()
        # Images to reconstruct during training
        self.input_imgs, self.label_imgs = imgs
        # Only save those images every N epochs (otherwise tensorboard gets quite large)
        self.every_n_epochs = every_n_epochs

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            # Reconstruct images
            input_imgs = self.input_imgs.to(pl_module.device)
            label_imgs = self.label_imgs.to(pl_module.device)
            with torch.no_grad():
                pl_module.eval()
                reconst_imgs = pl_module(input_imgs)
                pl_module.train()
            # Plot and add to tensorboard
            imgs = torch.stack(
                [input_imgs, reconst_imgs, label_imgs], dim=1).flatten(0, 1)
            grid = torchvision.utils.make_grid(
                imgs, nrow=3, normalize=True, range=(-1, 1))
            trainer.logger.experiment.add_image(
                "Reconstructions", grid, global_step=trainer.global_step)


def compare_imgs(img1, img2, title_prefix=""):
    # Calculate MSE loss between both images
    loss = F.mse_loss(img1, img2, reduction="sum")
    # Plot images for visual comparison
    grid = torchvision.utils.make_grid(torch.stack(
        [img1, img2], dim=0), nrow=2, normalize=True, range=(-1, 1))
    grid = grid.permute(1, 2, 0)
    plt.figure(figsize=(4, 2))
    plt.title(f"{title_prefix} Loss: {loss.item():4.2f}")
    plt.imshow(grid)
    plt.axis("off")
    plt.show()


def visualize_reconstructions(model, input_imgs):
    # Reconstruct images
    model.eval()
    with torch.no_grad():
        reconst_imgs = model(input_imgs.to(model.device))
    reconst_imgs = reconst_imgs.cpu()

    # Plotting
    imgs = torch.stack([input_imgs, reconst_imgs], dim=1).flatten(0, 1)
    grid = torchvision.utils.make_grid(
        imgs, nrow=4, normalize=True, range=(-1, 1))
    grid = grid.permute(1, 2, 0)
    plt.figure(figsize=(7, 4.5))
    plt.title("Reconstructed from %i latents" % (model.hparams.latent_dim))
    plt.imshow(grid)
    plt.axis("off")
    plt.show()


def get_train_images(data_dir, num, transform=None, img_size=(224, 224)):
    transform = transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]) if not transform else transform

    samples = os.listdir(data_dir)

    train_images = []
    label_images = []
    pbar = tqdm(total=num, desc="Building samples for logging")
    i = 0
    while len(train_images) <= num:
        if samples[i].endswith('-input.png'):
            train_images.append(transform(Image.open(
                os.path.join(
                    data_dir,
                    samples[i]
                ))))
            label_images.append(transform(Image.open(
                os.path.join(
                    data_dir,
                    samples[i].replace('-input', '-label1')
                ))))
            pbar.update(1)
        i += 1
    return torch.stack(train_images, dim=0), torch.stack(label_images, dim=0)


def generate_data_splits(data_dir_path, output_dir_path, train_split=0.7, validation_split=0.2, test_split=0.1):
    if not os.path.isdir(data_dir_path):
        raise NotADirectoryError(
            "The data directory path provided is not valid!")

    splits = round(
        sum([train_split, validation_split, test_split]), 3)
    if splits != 1:
        raise InvalidSplitsError(
            f"The provided data split percentages {splits} needs to add up to 1.0")
    split_dict = {'train': [], 'validation': [], 'test': []}
    input_files = [filename for filename in os.listdir(
        data_dir_path) if '-input' in filename]
    num_samples = len(input_files)
    random.shuffle(input_files)
    train_size = int(num_samples * train_split)
    validation_size = int(num_samples * validation_split)
    test_size = int(num_samples * test_split)
    train_size += int(num_samples - (train_size +
                                     validation_size + test_size))
    for filename in input_files:
        label1 = filename.replace('-input', '-label1')
        label2 = filename.replace('-input', '-label2')

        if train_size > 0:
            split_dict['train'].extend([filename, label1, label2])
            train_size -= 1
        elif validation_size > 0:
            # goes to validation set
            split_dict['validation'].extend([filename, label1, label2])
            validation_size -= 1
        else:
            # goes to testing set
            split_dict['test'].extend([filename, label1, label2])

    with open(os.path.join(output_dir_path, 'train.csv'), 'w') as f:
        writer = csv.writer(f)
        for file_name in split_dict['train']:
            writer.writerow([file_name])
    with open(os.path.join(output_dir_path, 'val.csv'), 'w') as f:
        writer = csv.writer(f)
        for file_name in split_dict['validation']:
            writer.writerow([file_name])
    with open(os.path.join(output_dir_path, 'test.csv'), 'w') as f:
        writer = csv.writer(f)
        for file_name in split_dict['test']:
            writer.writerow([file_name])


def read_data_splits(split_file_path):
    if not os.path.isfile(split_file_path):
        raise FileNotFoundError("Please provide valid path to a file!")
    with open(split_file_path) as f:
        reader = csv.reader(f)
        file_to_split = list(reader)
    return [row[0] for row in file_to_split]


def predict(module, img):

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    inp = transform(img).unsqueeze(0)

    outp = module(inp)

    return transforms.ToPILImage()(outp[0])


if __name__ == "__main__":
    '''train_dataset = CIFAR10(root=DATASET_PATH, train=True,
                            transform=transform, download=True)
    for i in range(2):
        # Load example image
        img, _ = train_dataset[i]
        img_mean = img.mean(dim=[1, 2], keepdims=True)

        # Shift image by one pixel
        SHIFT = 1
        img_shifted = torch.roll(img, shifts=SHIFT, dims=1)
        img_shifted = torch.roll(img_shifted, shifts=SHIFT, dims=2)
        img_shifted[:, :1, :] = img_mean
        img_shifted[:, :, :1] = img_mean
        compare_imgs(img, img_shifted, "Shifted -")

        # Set half of the image to zero
        img_masked = img.clone()
        img_masked[:, : img_masked.shape[1] // 2, :] = img_mean
        compare_imgs(img, img_masked, "Masked -")
    '''
    parent_dir = os.path.dirname(os.getcwd())
    data_dir_path = os.path.join(parent_dir, 'data/SIR2')
    output_dir_path = os.path.join(parent_dir, 'splits/SIR2')
    generate_data_splits(data_dir_path, output_dir_path)
    print(read_data_splits(os.path.join(output_dir_path, 'train.csv')))
    print()
    print()
    print(read_data_splits(os.path.join(output_dir_path, 'val.csv')))
    print()
    print()
    print(read_data_splits(os.path.join(output_dir_path, 'test.csv')))
