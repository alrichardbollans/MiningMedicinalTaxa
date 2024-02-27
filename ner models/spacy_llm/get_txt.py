import re
import os
import json
import time
import requests
from dotenv import load_dotenv



load_dotenv()
print("CORE_API_KEY:", os.getenv('CORE_API_KEY'))

def download_json_from_coreid(coreid_list, known_issues, directory):
    """Download JSON files from a list of core IDs."""
    for coreid in coreid_list:
        outpath = os.path.join(directory, f"{coreid}.json")
        if not os.path.isfile(outpath) and coreid not in known_issues:
            try:
                headers = {"Authorization": "Bearer " + os.getenv('CORE_API_KEY')}
                response = requests.get(f"https://api.core.ac.uk/v3/outputs/{coreid}", headers=headers)
                if response.status_code == 200:
                    with open(outpath, 'w') as f:
                        json.dump(response.json(), f)
                    time.sleep(5)  # Respect API rate limits
                else:
                    print(f"Failed to download {coreid}: HTTP {response.status_code}")
                    time.sleep(1)
            except Exception as e:
                print(f"Error downloading {coreid}: {e}")
                time.sleep(1)

# Example usage:
coreid_list = ["80818116",
    "286439500",
    "360558516",
    "228197190",
    "268329601",
    "197989253",
    "4187756",
    "481470282",
    "161880242",
    "35321774"]  # Replace with actual core ID
known_issues = ["error_id_1", "error_id_2"]  # example for ID know to have issue (maybe remove this)
directory = '10_medicinal_hits_json'  # Store Json files

#download_json_from_coreid(coreid_list, known_issues, directory)



# Compile regexes once first
# Split by looking for an instance of simple_string (ignoring case) begins a line on its own (or with line numbers) followed by any amount of whitespace and then a new line
# Must use re.MULTILINE flag such that the pattern character '^' matches at the beginning of the string and at the beginning of each line (immediately following each newline)
_reference_regex = re.compile(r"^\s*\d*\s*References\s*\n", flags=re.IGNORECASE | re.MULTILINE)
_supp_regex = re.compile(r"^\s*\d*\s*Supplementary material\s*\n", flags=re.IGNORECASE | re.MULTILINE)
_conf_regex = re.compile(r"^\s*\d*\s*Conflict of interest\s*\n", flags=re.IGNORECASE | re.MULTILINE)
_ackno_regex = re.compile(r"^\s*\d*\s*Acknowledgments\s*\n", flags=re.IGNORECASE | re.MULTILINE)

def retrieve_text_before_phrase(given_text: str, my_regex, simple_string: str) -> str:
    if simple_string.lower() in given_text.lower():

        text_split = my_regex.split(given_text, maxsplit=1)  # This is the bottleneck

        pre_split = text_split[0]
        if len(text_split) > 1:
            # At most 1 split occurs, if there has been a split the remainder of the string is returned as the final element of the list.
            # if text after split point is longer than before the split, then revert to given text.
            post_split = text_split[1]
            if len(post_split) > len(pre_split):
                pre_split = given_text

        return pre_split
    else:

        return given_text


def clean_paper_text(paper: dict) -> str:
    text = paper['fullText']
    if text is None:
        return None

    # Split by looking for an instance of 'Supplementary material' (ignoring case)
    # begins a line on its own (followed by any amount of whitespace and then a new line)
    pre_reference = retrieve_text_before_phrase(text, _reference_regex, 'References')
    pre_supplementary = retrieve_text_before_phrase(pre_reference, _supp_regex, 'Supplementary material')
    pre_conflict = retrieve_text_before_phrase(pre_supplementary, _conf_regex, 'Conflict of interest')
    pre_acknowledgement = retrieve_text_before_phrase(pre_conflict, _ackno_regex, 'Acknowledgments')

    return pre_acknowledgement


def process_json_files(input_directory: str, output_directory: str):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for filename in os.listdir(input_directory):
        if filename.endswith(".json"):
            file_path = os.path.join(input_directory, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                paper = json.load(f)

            cleaned_text = clean_paper_text(paper)
            if cleaned_text is not None:
                output_file_path = os.path.join(output_directory, os.path.splitext(filename)[0] + '.txt')
                with open(output_file_path, 'w', encoding='utf-8') as f:
                    f.write(cleaned_text)


# Specify your directories here
input_directory = '10_medicinal_hits_json'
output_directory = '10_medicinal_hits'

process_json_files(input_directory, output_directory)
