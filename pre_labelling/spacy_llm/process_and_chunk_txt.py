import os
import sys
import nltk
import tiktoken
from nltk.tokenize import sent_tokenize
import pandas as pd
import shutil
import ast
import re


sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'pre_labelling', 'spacy_llm'))

from useful_string_methods import retrieve_text_before_phrase, remove_double_spaces_and_break_characters, \
    convert_nonascii_to_ascii

nltk.download('punkt')


# Function to count tokens using tiktoken
def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

# Function to process text files and divide them in chunks of 4000 tokens
def process_and_chunk_text(input_directory, output_directory, max_tokens=4000, encoding_name="cl100k_base"):
    os.makedirs(output_directory, exist_ok=True)

    for file in os.listdir(input_directory):
        if file.endswith('.txt'):
            file_path = os.path.join(input_directory, file)
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()

            text = retrieve_text_before_phrase(text, 'References')
            text = remove_double_spaces_and_break_characters(text)
            text = convert_nonascii_to_ascii(text)
            sentences = sent_tokenize(text)

            chunks = []
            current_chunk = []
            current_tokens = 0
            for sentence in sentences:
                sentence_tokens = num_tokens_from_string(sentence, encoding_name)
                if current_tokens + sentence_tokens <= max_tokens:
                    current_chunk.append(sentence)
                    current_tokens += sentence_tokens
                else:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = [sentence]
                    current_tokens = sentence_tokens

            if current_chunk:
                chunks.append(' '.join(current_chunk))

            base_filename = os.path.splitext(os.path.basename(file_path))[0]
            for i, chunk in enumerate(chunks):
                output_file = os.path.join(output_directory, f'{base_filename}.txt_chunk_{i}.txt')
                with open(output_file, 'w', encoding='utf-8') as out:
                    out.write(chunk)

# Functions to select chunks that contain at least on keyword and save them in a separate folder
def aggregate_keywords(csv_path, subset_size):
    df = pd.read_csv(csv_path)
    df['aggregated_keywords'] = df.apply(
        lambda row: safe_eval_list(row['plant_species_binomials_counts']) + safe_eval_list(
            row['fungi_species_binomials_counts']), axis=1)

    if subset_size > len(df):
        subset_size = len(df)  # Ensure we don't exceed the DataFrame's length
    df_limited = df.iloc[:subset_size]

    corpus_keywords = pd.Series(df_limited.aggregated_keywords.values, index=df_limited.corpusid.astype(str)).to_dict()
    return corpus_keywords


def safe_eval_list(val):
    try:
        return ast.literal_eval(val) if isinstance(val, str) and val.strip() != '' else []
    except ValueError:
        return []

def extract_corpus_id(filename):
    return filename.split('.txt')[0]


def contains_keyword(content, keywords, filename, log_list):
    content_lower = content.lower()
    for keyword in keywords:
        regex_pattern = r'\b' + re.escape(keyword.lower()) + r'\b(\s*\([^)]*\))?'
        if re.search(regex_pattern, content_lower):
            print(f"Match found for keyword: '{keyword}' in file: {filename}")
            log_list.append({'chunk_id': filename, 'matching_keyword': keyword})
            return True
    return False


def select_chunks(csv_path, input_directory, output_directory, subset_size, log_path):
    """Select chunks containing keywords"""

    os.makedirs(output_directory, exist_ok=True)

    corpus_keywords = aggregate_keywords(csv_path, subset_size)

    print(f"Loaded keywords: {corpus_keywords}")

    matching_keywords_log = []

    for filename in os.listdir(input_directory):
        if filename.endswith(".txt") and "_chunk_" in filename:
            print(f"Processing file: {filename}")
            corpus_id = extract_corpus_id(filename)
            print(f"Extracted corpus ID: {corpus_id}")
            if corpus_id in corpus_keywords:
                print(f"Corpus ID {corpus_id} found in keywords")
                with open(os.path.join(input_directory, filename), 'r', encoding='utf-8') as file:
                    content = file.read()
                    if contains_keyword(content, corpus_keywords[corpus_id], filename, matching_keywords_log):
                        print(f"Keyword found in file: {filename}")
                        shutil.copy(os.path.join(input_directory, filename),
                                    os.path.join(output_directory, filename))
                    else:
                        print(f"No keyword found in file: {filename}")
            else:
                print(f"Corpus ID {corpus_id} not in keywords")

    if  matching_keywords_log:
        matching_keywords_df = pd.DataFrame(matching_keywords_log)
        matching_keywords_df.to_csv(os.path.join(log_path, 'matching_keywords_log.csv'), index=False)
        print("Matching keywords log saved to CSV.")


if __name__ == "__main__":
    main()