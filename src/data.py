import os
import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
from utils import read_data_splits


class ReflectionDataset(Dataset):
    '''
    Dataset class for loading reflection images from dataset

    Args:
        data_dir (str): Root directory of the dataset.
        split_file (str): file housing the split.
        transform (callable, Default: None): A function/transform that takes in an PIL image and returns a transformed version.
        img_size (tuple)(len: 2)(Default: (224, 224)): Size to resize all images to (img_size, img_size)
    Returns:
        dict: A dictionary containing 'input', 'label1', and 'label2'. The values are the transformed input image and corresponding label images.
    '''

    def __init__(self, data_dir, split_file, transform=None, img_size=(224, 224)):
        self.data_dir = data_dir
        self.split_file = split_file
        self.filenames = [filename for filename in read_data_splits(
            split_file) if filename.endswith('-input.png')]
        self.transform = transforms.Compose(
            [transforms.Resize(img_size), transforms.ToTensor()]) if not transform else transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        input_filename = os.path.join(self.data_dir, self.filenames[idx])
        label1_filename = input_filename.replace('-input', '-label1')
        label2_filename = input_filename.replace('-input', '-label2')

        try:
            # create the file images from the file names
            input_image = Image.open(input_filename)
            label1_image = Image.open(label1_filename)
            label2_image = Image.open(label2_filename)

            # Apply the transforms
            transformed_input = self.transform(input_image)
            transformed_label1 = self.transform(label1_image)
            transformed_label2 = self.transform(label2_image)

            # return the triplet
            return (transformed_input, transformed_label1, transformed_label2)
        except:
            return None


class ReflectionDataModule(pl.LightningDataModule):
    '''
    A PyTorch Lightning DataModule for loading reflection images and the label images associated with it

    Arguments:
        data_dir (str): The directory path where the dataset will be stored.
        gdrive_folder_url (str): The Google Drive folder URL that contains the dataset
        train_split (float)(Default: 0.7): The percentage of data to use for the training set. 
        validation_split (float)(Default: 0.2): The percentage of data to used for the validation set.
        test_split (float)(Default: 0.1): The percentage of data to used for the test set.
        batch_size (int)(Default: 32): The batch size to use for the data loaders.
        num_workers (int)(Default: 4): The number of workers to use for the data loaders.
    '''

    def __init__(self, split_dir, data_dir, batch_size=32, num_workers=4):
        super().__init__()
        self.data_dir = data_dir
        self.split_dir = split_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self):
        '''
        Performs the initialization of the dataset
        '''
        self.train_dataset = ReflectionDataset(
            self.data_dir, os.path.join(self.split_sir, "train.csv"))
        self.val_dataset = ReflectionDataset(
            self.data_dir, os.path.join(self.split_sir, "val.csv"))
        self.test_dataset = ReflectionDataset(
            self.data_dir, os.path.join(self.split_sir, "test.csv"))

    def collate_fn(self, batch):
        batch = list(filter(lambda x: x is not None, batch))
        return torch.utils.data.dataloader.default_collate(batch)

    def train_dataloader(self):
        '''
        Returns Lightning data loader for the training dataset
        '''
        return DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        '''Returns Lightning data loader for the training dataset'''
        return DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn, num_workers=self.num_workers)

    def test_dataloader(self):
        '''Returns Lightning data loader for the training dataset'''
        return DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn, num_workers=self.num_workers)


class InvalidSplitsError(Exception):
    pass


# Shows how to use the Reflection Data Module
if __name__ == "__main__":
    # This is where the data would be stored
    data_dir_path = os.path.join(os.path.dirname(os.getcwd()), 'data')
    split_dir_path = os.path.join(os.path.dirname(os.getcwd()), 'splits')
    data_module = ReflectionDataModule(
        split_dir=split_dir_path, data_dir=data_dir_path)

    # Downloads the Data
    data_module.prepare_data()

    # Sets up the splits of the data
    data_module.setup()

    # Get the dataloaders for train, split, test
    train_dataloader = data_module.train_dataloader()
    val_dataloader = data_module.val_dataloader()
    test_dataloader = data_module.test_dataloader()
    print(len(train_dataloader.dataset))
    print(len(test_dataloader.dataset))
    print(len(val_dataloader.dataset))
    # Example of how to loop through each batch of the train dataloader
    for batch in train_dataloader:
        # how to access the input image (this has the reflections)
        print(len(batch[0]), '\n')
        print(len(batch[1]), '\n')  # how to access the first image
        # how to access the second image which is the reflection in the input image
        print(len(batch[2]), '\n')
        print(batch)
        print('\n\n\n')
