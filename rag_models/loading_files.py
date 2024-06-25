from useful_string_methods import remove_double_spaces_and_break_characters
from langchain_text_splitters import TokenTextSplitter

def get_txt_from_file(txt_file: str):
    import os

    with open(os.path.join(txt_file), "r", encoding="utf8") as f:
        text = f.read()

    out = remove_double_spaces_and_break_characters(text)

    return out


def read_file_and_chunk(txt_file: str, context_size:int) -> list:
    # chunking specific to models: https://python.langchain.com/v0.1/docs/use_cases/extraction/how_to/handle_long_text/
    text = get_txt_from_file(txt_file)
    text_splitter = TokenTextSplitter(
        # Controls the size of each chunk
        chunk_size=context_size,
        # Controls overlap between chunks
        chunk_overlap=20,
    )

    texts = text_splitter.split_text(text)

    return texts