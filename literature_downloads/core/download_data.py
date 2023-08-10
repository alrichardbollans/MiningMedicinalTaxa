import os

import dotenv

dotenv.load_dotenv()
CORE_API_KEY = os.environ['CORE_API_KEY']
headers = {
    'x-api-key': CORE_API_KEY,
}


def download_full_core_dataset():
    # dataset docs = https://core.ac.uk/documentation/dataset

    os.system('wget -c https://core.ac.uk/datasets/core_2022-03-11_dataset.tar.xz')


if __name__ == '__main__':
    download_full_core_dataset()
