import os.path
import time

import dotenv

dotenv.load_dotenv()
S2_API_KEY = os.environ['S2_API_KEY']
headers = {
    'x-api-key': S2_API_KEY,
}

# docs: https://api.semanticscholar.org/api-docs/datasets
sem_schol_project_path = os.path.join(os.environ.get('SCRATCH'), 'MedicinalPlantMining', 'literature_downloads', 'semantic_scholar')

sem_schol_download_path = os.path.join(sem_schol_project_path, 'downloads')
sem_schol_dataset_download_path = os.path.join(sem_schol_download_path, 'datasets')
_dirs = [sem_schol_download_path, sem_schol_dataset_download_path]
for d in _dirs:
    if not os.path.exists(d):
        os.mkdir(d)


def get_zip_file_for_s2orc_part(part: int):
    return os.path.join(sem_schol_dataset_download_path, "s2orc-part" + str(part) + ".jsonl.gz")


def get_unzipped_file_for_s2orc_part(part: int):
    return os.path.join(sem_schol_dataset_download_path, "s2orc-part" + str(part) + ".jsonl")

def get_zip_file_for_sem_paper_part(part: int):
    return os.path.join(sem_schol_dataset_download_path, "papers-part" + str(part) + ".jsonl.gz")

def download_fullset():
    import requests
    import urllib
    # Get info about the latest release
    latest_release = requests.get("http://api.semanticscholar.org/datasets/v1/release/latest").json()
    print(latest_release['README'])
    print(latest_release['release_id'])

    # Print names of datasets in the release
    print("\n".join(d['name'] for d in latest_release['datasets']))

    # Print README for one of the datasets
    print(latest_release['datasets'][2]['README'])

    for part in range(0, 30):
        # Get the s2orc dataset
        s2orc = requests.get("http://api.semanticscholar.org/datasets/v1/release/latest/dataset/s2orc",
                             headers={'x-api-key': S2_API_KEY}).json()
        print(part)
        # Download the part of the dataset
        urllib.request.urlretrieve(s2orc['files'][part],
                                   get_zip_file_for_s2orc_part(part))
        print(f'part {part} done')
        time.sleep(10)

def download_metadata():
    import requests
    import urllib
    for part in range(0, 30):
        # Get info about the papers
        # Get info about the papers dataset
        papers = requests.get("http://api.semanticscholar.org/datasets/v1/release/latest/dataset/papers",
                              headers={'X-API-KEY': os.getenv("S2_API_KEY")}).json()

        # Download the first part of the dataset
        urllib.request.urlretrieve(papers['files'][part], get_zip_file_for_sem_paper_part(part))


if __name__ == '__main__':
    # Download then check they're all there
    # download_fullset()
    download_metadata()
