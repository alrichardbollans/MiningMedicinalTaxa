import os.path
from pdfminer.high_level import extract_text

from useful_string_methods import remove_double_spaces_and_break_characters, convert_nonascii_to_ascii


def get_paragraphs_from_pdf(pdf_file: str):
    text = extract_text(pdf_file)
    print(text)
    print(repr(text))
    special_paragraph_placeholder = 'xyzxyzplaceholder'
    placeholder = text.replace('\n\n', special_paragraph_placeholder)
    # Make line breaks more readable
    placeholder = placeholder.replace('-\n', '')
    placeholder = placeholder.replace('\n', ' ')
    paragraphs = placeholder.split(special_paragraph_placeholder)
    return paragraphs


def split_text_by_limit(text: str, limit: int):
    import textwrap
    lines = textwrap.wrap(text, limit)
    return lines


def get_chunks_from_pdf(pdf_file: str, chunk_character_limit: int):
    paragraphs = get_paragraphs_from_pdf(pdf_file)

    paragraph_character_limit = chunk_character_limit - 2  # Edit chunk limit to account for addition of \n\n
    shortened_paragraphs = []
    for paragraph in paragraphs:
        text = remove_double_spaces_and_break_characters(paragraph)
        text = convert_nonascii_to_ascii(text)

        if len(text) <= paragraph_character_limit:
            shortened_paragraphs.append(text)
        else:
            broken_paragraphs = split_text_by_limit(text, paragraph_character_limit)
            shortened_paragraphs.extend(broken_paragraphs)

    chunks = []
    chunk = ''
    for index, paragraph in enumerate(shortened_paragraphs):
        if len(chunk) + len(paragraph) <= paragraph_character_limit:
            if index < len(shortened_paragraphs) - 1:
                chunk += paragraph + '\n\n'
            else:
                chunk += paragraph
                chunks.append(chunk)
        else:
            chunks.append(chunk)
            chunk = paragraph

    return chunks


if __name__ == '__main__':
    _inputs_path = 'example_inputs'
    get_chunks_from_pdf(os.path.join(_inputs_path, 'normal.pdf'), 20000)
