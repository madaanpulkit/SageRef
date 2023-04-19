import requests
import os
from bs4 import BeautifulSoup
import shutil
import threading
import argparse
import sys
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from src.model import Autoencoder
from src.data import ReflectionDataModule
from src.utils import GenerateCallback
from src.utils import get_train_images


def download_data_from_google_drive(data_dir):
    '''
    Downloads the dataset from the specified Google Drive folder URL in the constructor to the 
    specified data directory.
    '''
    folder_ids = {'CEILNET': '1kNnCS58dCcHsZVS2dDyDmxugcxLSuPF5',
                  'SIR2': '1c8BKPk1y6aJ84EBIeQEevpdFs-AjZxIb'}
    need_to_download = check_for_data_folder_downloads(data_dir)

    for folder_name in need_to_download:
        folder_path = os.path.join(data_dir, folder_name)
        download_folder(folder_path, folder_ids[folder_name])


def check_for_data_folder_downloads(data_dir):
    '''
        Determines the data folders that are missing and returns a list with their names.
        Args:
            data_dir (string): directory path to where the datasets are stored
        Returns:
            list: contains names of the datasets
    '''
    folder_names = ['CEILNET', 'SIR2']
    def validate_dir(path): return os.path.isdir(path) and os.listdir(path)
    need_to_download = [dirname for dirname in folder_names if not validate_dir(
        os.path.join(data_dir, dirname))]
    return need_to_download


def download_folder(folder_path, folder_id, batch_size=100, num_threads=20):
    '''
        Downloads a google drive folder by utilizing threads.
        Args:
            folder_path (string): directory path to where folder will be downloaded
            folder_id (string): google drive folder id
            batch_size (int): number of files to be grouped together and assigned to a thread
            num_threads (int)
        Returns:
            list: contains names of the datasets

    '''
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    url = f'https://drive.google.com/embeddedfolderview?id={folder_id}#list'
    soup = BeautifulSoup(requests.get(
        url, verify=True).content, 'html.parser')
    flip_entries = soup.find_all(class_='flip-entry')

    for i in range(0, len(flip_entries), batch_size):
        entries = flip_entries[i: i + batch_size]
        batches = [entries[j:j+num_threads]
                   for j in range(0, len(entries), num_threads)]

        for batch in batches:
            threads = []
            for entry in batch:
                file_id = entry.get('id').replace('entry-', '')
                title = entry.find(
                    class_='flip-entry-title').encode_contents().decode()
                file_dir = os.path.join(folder_path, title)
                t = threading.Thread(
                    target=download_file, args=(file_id, file_dir))
                t.start()
                threads.append(t)

            for t in threads:
                t.join()


def download_file(file_id, file_dir):
    '''
    Downloads a google drive file
        Args:
            file_id (string): google drive file id
            file_dir (string): path to where file will be saved
    '''
    r = requests.get(
        f'https://drive.google.com/uc?export=download&id={file_id}', stream=True)
    with open(file_dir, 'wb') as f:
        shutil.copyfileobj(r.raw, f)


def main(args):
    # Create data dir
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)

    # Download data
    if args.mode == 'download' and check_for_data_folder_downloads(args.data_dir):
        print(f"Downloading data to {args.data_dir}")
        download_data_from_google_drive(
            data_dir=args.data_dir
        )
        print(f"Data downloaded at {args.data_dir}")
        sys.exit(0)

    # Create out dir
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    module = Autoencoder(args.latent_dim, args.learning_rate)
    datamodule = ReflectionDataModule(
        args.split_dir, args.data_dir, args.batch_size)
    callbacks = [
        ModelCheckpoint(dirpath=args.out_dir),
        ModelCheckpoint(monitor="val/loss",
                        dirpath=args.out_dir, filename="best"),
        GenerateCallback(get_train_images(
            args.data_dir, 4), every_n_epochs=10),
        LearningRateMonitor("epoch")]
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=[args.gpu],
        callbacks=callbacks,
        default_root_dir=args.out_dir,
        detect_anomaly=True)

    if args.mode == "train":
        trainer.fit(module, datamodule)
        trainer.test(module, datamodule)
    elif args.mode == "test":
        module.load_from_checkpoint(args.ckpt_path)
        trainer.test(module, datamodule)
    elif args.mode == "predict":
        module.load_from_checkpoint(args.ckpt_path)
        trainer.predict(module, datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='run the relection removal experiment'
    )
    parser.add_argument(
        '--gpu',
        type=int,
        required=True,
        help='gpu id'
    )
    parser.add_argument(
        '--mode',
        required=True,
        choices=['train', 'test', 'predict', 'download'],
        help='mode: [train, test, predict, 'download']'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=2,
        help='number of training epochs'
    )
    parser.add_argument(
        '--latent_dim',
        type=int,
        default=64,
        help='feature dimensions for latent space'
    )
    parser.add_argument(
        '--out_dir',
        default=os.path.join(os.getcwd(), 'out'),
        help='output directory'
    )
    parser.add_argument(
        '--data_dir',
        default=os.path.join(os.getcwd(), 'data'),
        help='data directory'
    )
    parser.add_argument(
        '--split_dir',
        default=os.path.join(os.getcwd(), 'splits'),
        help='data directory'
    )
    parser.add_argument(
        '--ckpt_path',
        help='checkpoint path'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='batch size for training'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-3,
        help='learning rate')

    args = parser.parse_args()

    main(args)
