import requests
import os
from bs4 import BeautifulSoup
import shutil
import threading
import argparse
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from src.model import Autoencoder
from src.data import ReflectionDataModule


def download_data_from_google_drive():
    '''
    Downloads the dataset from the specified Google Drive folder URL in the constructor to the 
    specified data directory.
    '''
    data_dir = os.path.join(os.getcwd(), 'data')
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)

    url = 'https://drive.google.com/embeddedfolderview?id=1kNnCS58dCcHsZVS2dDyDmxugcxLSuPF5#list'
    soup = BeautifulSoup(requests.get(url).content, 'html.parser')
    flip_entries = soup.find_all(class_='flip-entry')
    threads = []
    for entry in flip_entries:
        file_id = entry.get('id').replace('entry-', '')
        title = entry.find(
            class_='flip-entry-title').encode_contents().decode()
        file_dir = os.path.join(data_dir, title)
        thread = threading.Thread(
            target=download_file, args=(file_id, file_dir))
        thread.start()
        threads.append(thread)

    # Wait for all threads to finish
    for thread in threads:
        thread.join()


def download_file(file_id, file_dir):
    r = requests.get(
        f'https://drive.google.com/uc?export=download&id={file_id}', stream=True)
    with open(file_dir, 'wb') as f:
        shutil.copyfileobj(r.raw, f)


def main(args):
    # Create data dir
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)
    # Download data
    if not os.listdir(args.data_dir):
        download_data_from_google_drive(
            data_dir=args.data_dir, data_url="https://drive.google.com/drive/folders/1kNnCS58dCcHsZVS2dDyDmxugcxLSuPF5?usp=share_link")
    # Create out dir
    if not os.path.exists(args.data_dir):
        os.makedirs(args.out_dir)

    module = Autoencoder(args.latent_dim, args.learning_rate)
    datamodule = ReflectionDataModule(
        args.data_dir, args.split_dir, args.batch_size)
    callbacks = [ModelCheckpoint(monitor="val_loss", dirpath=args.out_dir)]
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=[args.gpu],
        callbacks=callbacks,
        enable_checkpointing=True,
        default_root_dir=args.out_dir,
        detect_anomaly=True)

    if args.mode == "train":
        trainer.fit(module, datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='run the relection removal experiment'
        )
    parser.add_argument(
        '--gpu', 
        type=int, 
        help='gpu id'
        )
    parser.add_argument(
        '--mode', 
        required=True, 
        choices=['train', 'eval', 'predict'], 
        help='mode: [train, eval, predict]')
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='number of training epochs')
    parser.add_argument(
        '--latent_dim',
        type=int,
        default=512,
        help='feature dimensions for mcr2 projection')
    parser.add_argument(
        '--out_dir',
        default=os.path.join(os.getcwd(), 'out'),
        help='output directory')
    parser.add_argument(
        '--data_dir',
        default=os.path.join(os.getcwd(), 'data'),
        help='data directory')
    parser.add_argument(
        '--split_dir',
        default=os.path.join(os.getcwd(), 'splits'),
        help='data directory')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='batch size for training')
    parser.add_argument(
        '--learning_rate',
        type=int,
        default=1e-3,
        help='learning rate')

    args = parser.parse_args()

    main(args)
