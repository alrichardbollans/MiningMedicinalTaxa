import os
import nltk
import tiktoken
from nltk.tokenize import sent_tokenize

nltk.download('punkt')

# Function to count tokens using tiktoken
def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def process_text_file(file_path, max_tokens=4000, encoding_name="cl100k_base"):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

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

input_folder = '50_medicinal_hits'
output_folder = 'preprocessed'
os.makedirs(output_folder, exist_ok=True)

for file in os.listdir(input_folder):
    if file.endswith('.txt'):
        file_path = os.path.join(input_folder, file)
        chunks = process_text_file(file_path)
        for i, chunk in enumerate(chunks):
            output_file = os.path.join(output_folder, f'{file}_chunk_{i}.txt')
            with open(output_file, 'w', encoding='utf-8') as out:
                out.write(chunk)
