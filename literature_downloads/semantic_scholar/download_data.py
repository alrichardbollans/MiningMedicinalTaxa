# Shows how to download and inspect data in the sample datasets
# which are much smaller than the full datasets.
import json
import os.path

from literature_downloads.semantic_scholar import download_path

dataset_download_path = os.path.join(download_path, 'datasets')
if not os.path.exists(dataset_download_path):
    os.mkdir(dataset_download_path)


def convert_corpus_id_to_paper_id(corpus_id) -> str:
    docs = [json.loads(l) for l in open("samples/paper-ids/paper-ids-sample.jsonl", "r").readlines()]
    shas = [x['sha'] for x in docs if x['corpusid'] == corpus_id]
    return shas


def look_at_sample():
    # subprocess.check_call("bash get_sample_files.sh", shell=True)

    # want to use paper id to get OCR data
    # use paper-ids to map between corpus ids and paperids

    # S2ORC
    docs = [json.loads(l) for l in open("samples/s2orc/s2orc-sample.jsonl", "r").readlines()]
    for paper in docs:
        text = paper['content']['text']
        annotations = {k: json.loads(v) for k, v in paper['content']['annotations'].items() if v}

        for a in annotations['paragraph'][:10]:
            print(a)
        for a in annotations['bibref'][:10]:
            print(a)
        for a in annotations['bibentry'][:10]:
            print(a)

        def text_of(type):
            return [text[a['start']:a['end']] for a in annotations[type]]

        paragraph_text = text_of('paragraph')
        paper_id = convert_corpus_id_to_paper_id(paper['corpusid'])
        print('\n\n'.join(text_of('paragraph')[:3]))
        print()


def download_fullset():
    import requests
    import urllib
    import os

    # Get info about the latest release
    latest_release = requests.get("http://api.semanticscholar.org/datasets/v1/release/latest").json()
    print(latest_release['README'])
    print(latest_release['release_id'])

    # Get info about past releases
    dataset_ids = requests.get("http://api.semanticscholar.org/datasets/v1/release").json()

    # Print names of datasets in the release
    print("\n".join(d['name'] for d in latest_release['datasets']))

    # Print README for one of the datasets
    print(latest_release['datasets'][2]['README'])

    # Get info about the papers dataset
    s2orc = requests.get("http://api.semanticscholar.org/datasets/v1/release/latest/dataset/s2orc",
                         headers={'x-api-key': os.getenv("S2_API_KEY")}).json()

    # Download the first part of the dataset
    urllib.request.urlretrieve(s2orc['files'][0],
                               os.path.join(dataset_download_path, "s2orc-part0.jsonl.gz"))

    # Get info about the papers dataset
    paper_ids = requests.get("http://api.semanticscholar.org/datasets/v1/release/latest/dataset/paper-ids",
                             headers={'x-api-key': os.getenv("S2_API_KEY")}).json()

    # Download the first part of the dataset
    urllib.request.urlretrieve(paper_ids['files'][0],
                               os.path.join(dataset_download_path, "paper-ids-part0.jsonl.gz"))


if __name__ == '__main__':
    # look_at_sample()
    download_fullset()
