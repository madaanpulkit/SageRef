import gdown
import os

def download_data_from_google_drive(data_dir, data_url):
    '''
    Downloads the dataset from the specified Google Drive folder URL in the constructor to the 
    specified data directory.
    '''

    if not os.path.isdir(data_dir):
        gdown.download_folder(url=data_url, output=data_dir,
                                quiet=True, use_cookies=False, remaining_ok=True)

if __name__ == "__main__":
    data_dir_path = os.path.join(os.getcwd(), 'data')
    download_data_from_google_drive(data_dir=data_dir_path, data_url="https://drive.google.com/drive/folders/1kNnCS58dCcHsZVS2dDyDmxugcxLSuPF5?usp=share_link")