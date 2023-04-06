import requests
import os
from bs4 import BeautifulSoup
import shutil
import threading

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
        title = entry.find(class_='flip-entry-title').encode_contents().decode()
        file_dir = os.path.join(data_dir, title)
        thread = threading.Thread(target=download_file, args=(file_id, file_dir))
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


if __name__ == "__main__":
    download_data_from_google_drive()