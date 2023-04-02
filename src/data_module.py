import os

import gdown
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image


class ReflectionDataset(Dataset):
    '''
    Dataset class for loading reflection images from dataset

    Args:
        root_dir (str): Root directory of the dataset.
        transform (callable, Default: None): A function/transform that takes in an PIL image and returns a transformed version.
        img_size (tuple)(len: 2)(Default: (224, 224)): Size to resize all images to (img_size, img_size)
    Returns:
        dict: A dictionary containing 'input', 'label1', and 'label2'. The values are the transformed input image and corresponding label images.
    '''

    def __init__(self, root_dir, transform=None, img_size = (224, 224)):
        self.root_dir = root_dir
        self.filenames = [filename for filename in os.listdir(
            root_dir) if filename.endswith('-input.png')]
        self.transform = transforms.Compose(
            [transforms.Resize(img_size), transforms.ToTensor()]) if not transform else transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        input_filename = os.path.join(self.root_dir, self.filenames[idx])
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
            return {'input': transformed_input, 'label1': transformed_label1, 'label2': transformed_label2}
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

    def __init__(self, data_dir, gdrive_folder_url, train_split=0.7, validation_split=0.2, test_split=0.1, batch_size=32, num_workers=4):
        super().__init__()
        self.data_dir = data_dir
        self.data_url = gdrive_folder_url
        self.train_split = round(train_split, 3)
        self.validation_split = round(validation_split, 3)
        self.test_split = round(test_split, 3)
        self.batch_size = batch_size
        self.num_workers = num_workers
        splits = round(
            sum([self.train_split, self.validation_split, self.test_split]), 3)
        if splits != 1:
            raise InvalidSplitsError(
                f"The provided data split percentages {splits} needs to add up to 1.0")


    def prepare_data(self):
        '''
        Downloads the dataset from the specified Google Drive folder URL in the constructor to the 
        specified data directory.
        '''

        if not os.path.isdir(self.data_dir):
            gdown.download_folder(url=self.data_url, output=self.data_dir,
                                  quiet=True, use_cookies=False, remaining_ok=True)


    def setup(self):
        '''
        Performs the dataset splits into training, validation and testing sets
        '''
        dataset = ReflectionDataset(root_dir=self.data_dir)
        self.num_samples = len(dataset)
        self.train_size = int(self.num_samples * self.train_split)
        self.test_size = int(self.num_samples * self.test_split)
        self.validation_size = int(self.num_samples * self.validation_split)
        self.train_size += int(self.num_samples - (self.train_size +
                               self.validation_size + self.test_size))

        # Splits randomly by the percentages of splits
        self.train_dataset, self.validation_dataset, self.test_dataset = torch.utils.data.random_split(
            dataset, [self.train_size, self.validation_size, self.test_size])

    def collate_fn(self, batch):
        batch = list(filter(lambda x: x is not None, batch))
        return torch.utils.data.dataloader.default_collate(batch)


    def train_dataloader(self):
        '''
        Returns Lightning data loader for the training dataset
        '''
        return DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn= self.collate_fn, shuffle=True, num_workers=self.num_workers)


    def validation_dataloader(self):
        '''Returns Lightning data loader for the training dataset'''
        return DataLoader(self.validation_dataset, batch_size=self.batch_size, collate_fn= self.collate_fn, num_workers=self.num_workers)


    def test_dataloader(self):
        '''Returns Lightning data loader for the training dataset'''
        return DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn= self.collate_fn, num_workers=self.num_workers)


class InvalidSplitsError(Exception):
    pass


# Shows how to use the Reflection Data Module
if __name__ == "__main__":
    # This is where the data would be stored
    data_dir_path = os.path.join(os.getcwd(), 'data')
    data_module = ReflectionDataModule(
        data_dir=data_dir_path, gdrive_folder_url='https://drive.google.com/drive/folders/1kNnCS58dCcHsZVS2dDyDmxugcxLSuPF5?usp=share_link')

    # Downloads the Data
    data_module.prepare_data()

    # Sets up the splits of the data
    data_module.setup()

    # Get the dataloaders for train, split, test
    train_dataloader = data_module.train_dataloader()
    validation_dataloader = data_module.validation_dataloader()
    test_dataloader = data_module.test_dataloader()

    # Example of how to loop through each batch of the train dataloader
    for batch in train_dataloader:
        # how to access the input image (this has the reflections)
        print(len(batch['input']), '\n')
        print(len(batch['label1']), '\n')  # how to access the first image
        # how to access the second image which is the reflection in the input image
        print(len(batch['label2']), '\n')
        print('\n\n\n')