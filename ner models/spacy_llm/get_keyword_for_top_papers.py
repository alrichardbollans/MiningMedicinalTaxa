import os
import pandas as pd
import shutil
import ast


df = pd.read_csv('medicinals_top_10000.csv')

# Keywords column names  are 'plant_species_binomials_counts' and 'fungi_species_binomials_counts'
df['aggregated_keywords'] = df['plant_species_binomials_counts'].astype(str) + ', ' + df[
    'fungi_species_binomials_counts'].astype(str)

# Limit to first 10 rows and then create the keywords dictionary
df_limited = df.iloc[:10]
corpus_keywords = pd.Series(df_limited.aggregated_keywords.values, index=df_limited.corpusid.astype(str)).to_dict()
# Remove nan values derived from combining keywords
corpus_keywords_cleaned = {corpus_id: keywords.replace(', nan', '').replace('nan, ', '').replace('nan', '') for corpus_id, keywords in corpus_keywords.items()}


# Save corpus_keywords to a CSV file
corpus_keywords_df = pd.DataFrame(list(corpus_keywords_cleaned.items()), columns=['corpusid', 'keywords'])
# corpus_keywords_df.head(10).to_csv('corpus_keywords.csv', index=False)
#corpus_keywords_df.to_csv('corpus_keywords.csv', index = False)

# Define input dir
preprocessed_dir = 'preprocessed'
# Define output dir
selected_preprocessed_dir = 'selected_preprocessed'
if not os.path.exists(selected_preprocessed_dir):
    os.makedirs(selected_preprocessed_dir)

# Function to extract corpusid from filename
def extract_corpus_id(filename):
    # Extracts the numeric part before the first ".txt"
    return filename.split('.txt')[0]

# Iterate through files and check for each corpus ID
for filename in os.listdir(preprocessed_dir):
    corpus_id = extract_corpus_id(filename)
    # Check if the extracted corpus ID is in the corpus_keywords dictionary
    if corpus_id in corpus_keywords_cleaned:
        print(f"Corpus ID {corpus_id} found in corpus_keywords with keywords: {corpus_keywords_cleaned[corpus_id]}")
    else:
        print(f"Corpus ID {corpus_id} not found in corpus_keywords.")

def contains_keyword(content, keywords, filename, log_list):
    content_lower = content.lower()
    for keyword in keywords:
        # Debug: Print the keyword before checking for a match
        print(f"Checking for keyword: '{keyword}'")
        if keyword.lower() in content_lower:
            # Debug: Print what it's about to log
            print(f"Match found for keyword: '{keyword}' in file: {filename}")
            log_list.append({'chunk_id': filename, 'matching_keyword': keyword})
            return True
    return False


# Example: Print the first 5 key-value pairs from the cleaned dictionary
for i, (corpus_id, keywords) in enumerate(corpus_keywords_cleaned.items()):
    if i < 11: print(corpus_id, ":", keywords)

# Prepare a list to log matching keywords
matching_keywords_log = []

# Iterate through each file in the preprocessed directory
for filename in os.listdir(preprocessed_dir):
    if filename.endswith(".txt") and "_chunk_" in filename:
        corpus_id = extract_corpus_id(filename)  # Extract corpus ID from filename
        print(f"Processing file: {filename} with corpus ID: {corpus_id}")
        if corpus_id in corpus_keywords_cleaned:
            # Read the content of the file
            with open(os.path.join(preprocessed_dir, filename), 'r', encoding='utf-8') as file:
                content = file.read()
                actual_list_of_keywords = ast.literal_eval(corpus_keywords_cleaned[corpus_id])
                # Check for any keyword in the content
                if contains_keyword(content, actual_list_of_keywords, filename, matching_keywords_log):
                    # If a keyword is found, save the file in the selected_preprocessed directory
                    shutil.copy(os.path.join(preprocessed_dir, filename),
                                os.path.join(selected_preprocessed_dir, filename))
                    print(f"Keyword found in file: {filename}. File copied.")  # Print when a file is copied
                else:
                    print(f"No matching keyword found in file: {filename}.")  # Print when no keyword is found

        # Convert the log list to a DataFrame and save it as CSV
        matching_keywords_df = pd.DataFrame(matching_keywords_log)
        matching_keywords_df.to_csv('matching_keywords_log.csv', index=False)
        print("Matching keywords log saved to CSV.")
