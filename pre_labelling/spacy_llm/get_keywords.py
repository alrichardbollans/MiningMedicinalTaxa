import os
import pandas as pd
import shutil
import ast
import re


def load_and_process_csv(csv_path, subset_size):
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


def get_chunks(csv_path, preprocessed_dir, selected_preprocessed_dir, subset_size=10):
    corpus_keywords = load_and_process_csv(csv_path, subset_size)

    if not os.path.exists(selected_preprocessed_dir):
        os.makedirs(selected_preprocessed_dir)

    matching_keywords_log = []

    for filename in os.listdir(preprocessed_dir):
        if filename.endswith(".txt") and "_chunk_" in filename:
            corpus_id = extract_corpus_id(filename)
            if corpus_id in corpus_keywords:
                with open(os.path.join(preprocessed_dir, filename), 'r', encoding='utf-8') as file:
                    content = file.read()
                    if contains_keyword(content, corpus_keywords[corpus_id], filename, matching_keywords_log):
                        shutil.copy(os.path.join(preprocessed_dir, filename),
                                    os.path.join(selected_preprocessed_dir, filename))

    if matching_keywords_log:
        matching_keywords_df = pd.DataFrame(matching_keywords_log)
        matching_keywords_df.to_csv(os.path.join(os.getcwd(), 'matching_keywords_log.csv'), index=False)
        print("Matching keywords log saved to CSV.")


if __name__ == "__main__":
    get_chunks()