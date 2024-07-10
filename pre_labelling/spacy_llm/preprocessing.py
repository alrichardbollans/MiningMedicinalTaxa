import sys
import os
import nltk
import tiktoken
from nltk.tokenize import sent_tokenize

# Adjust the system path to include the directory where your text processing functions are located
# This is assuming your current script is in the 'testing' directory and your utilities are in 'ner_models/spacy_llm'
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ner_models', 'spacy_llm'))

from useful_string_methods import (
    retrieve_text_before_phrase,
    remove_double_spaces_and_break_characters,
    convert_nonascii_to_ascii
)

nltk.download('punkt')

# Function to count tokens using tiktoken
def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def process_text_file(file_path, max_tokens=4000, encoding_name="cl100k_base"):
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

    return chunks

def process_files(input_folder, output_folder, max_tokens=4000, encoding_name="cl100k_base"):
    os.makedirs(output_folder, exist_ok=True)

    for file in os.listdir(input_folder):
        if file.endswith('.txt'):
            file_path = os.path.join(input_folder, file)
            chunks = process_text_file(file_path, max_tokens, encoding_name)
            for i, chunk in enumerate(chunks):
                output_file = os.path.join(output_folder, f'{file}_chunk_{i}.txt')
                with open(output_file, 'w', encoding='utf-8') as out:
                    out.write(chunk)

if __name__ == "__main__":
   main()