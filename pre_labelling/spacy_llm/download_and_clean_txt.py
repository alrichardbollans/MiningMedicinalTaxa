import json
import os
import time

import requests
from dotenv import load_dotenv

from useful_string_methods import retrieve_text_before_phrase

# TODO: Import _download_json_from_coreid and clean_text_paper functions directly from literature_downloads.core
load_dotenv()
CORE_API_KEY = os.getenv('CORE_API_KEY')


# Had to isolate _download_json_from_coreid and clean_paper_text as I got error if calling it directly from literature_downloads.core import (_download_json_from coreid)
def _download_json_from_coreid(coreid: str, outpath: str, known_issues: str):
    """ If file doesn't already exist and coreid not in list of known issues, download the core json to the outpath."""
    if not os.path.isfile(outpath):
        if coreid not in known_issues:
            headers = {"Authorization": "Bearer " + CORE_API_KEY}
            response = requests.get(f"https://api.core.ac.uk/v3/outputs/{coreid}", headers=headers)
            output_json = response.json()
            if response.status_code != 200:
                time.sleep(1)
                raise ValueError
            else:
                with open(outpath, 'w') as f:
                    json.dump(output_json, f)
                time.sleep(5)  # 10,000 tokens per day, maximum 10 per minute.
        else:
            raise ValueError


def clean_paper_text(paper: dict) -> str:
    """Clean paper text by removing sections after certain phrases."""
    text = paper['fullText']
    if text is None:
        return None

    pre_reference = retrieve_text_before_phrase(text,
                                                'References')  # This has been modified to work with retrieve_text_before_phrases stored in useful_string_methods
    pre_supplementary = retrieve_text_before_phrase(pre_reference, 'Supplementary material')
    pre_conflict = retrieve_text_before_phrase(pre_supplementary, 'Conflict of interest')
    pre_acknowledgement = retrieve_text_before_phrase(pre_conflict, 'Acknowledgments')

    return pre_acknowledgement


def process_and_save_text(coreid: str, known_issues: str, output_directory: str):
    """Download, process, and save JSON as text."""
    os.makedirs(output_directory, exist_ok=True)
    temp_json_path = os.path.join(output_directory, f"{coreid}_temp.json")
    _download_json_from_coreid(coreid, temp_json_path, known_issues)

    with open(temp_json_path, 'r') as file:
        paper_json = json.load(file)

    cleaned_text = clean_paper_text(paper_json)
    if cleaned_text is not None:
        output_file_path = os.path.join(output_directory, f"{coreid}.txt")
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_text)

    # Remove the temporary JSON file
    os.remove(temp_json_path)


if __name__ == "__main__":
    main()
