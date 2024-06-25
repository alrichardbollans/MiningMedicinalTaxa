import os

from langchain_anthropic import ChatAnthropic
from langchain_mistralai import ChatMistralAI
from langchain_openai import ChatOpenAI
from langchain_google_vertexai import ChatVertexAI

from rag_models.base_prompt_structure import standard_medicinal_prompt
from rag_models.loading_files import get_txt_from_file, read_file_and_chunk
from rag_models.structured_output_schema import deduplicate_and_standardise_output_taxa_lists, TaxaData


def query_a_model(model, text_file: str, context_window: int) -> TaxaData:
    text_chunks = read_file_and_chunk(text_file, context_window)
    extractor = standard_medicinal_prompt | model.with_structured_output(schema=TaxaData, include_raw=False)
    extractions = extractor.batch(
        [{"text": text} for text in text_chunks],
        {"max_concurrency": 1},  # limit the concurrency by passing max concurrency! Otherwise Requests rate limit exceeded
    )
    # output = extractor.invoke({'text': text})
    output = []

    for extraction in extractions:
        output.extend(extraction.taxa)

    deduplicated_extractions = deduplicate_and_standardise_output_taxa_lists(output)
    return deduplicated_extractions

def get_input_size_limit(total_context_window_k: int):
    # Output tokens so far is a tiny fraction, so allow 5% of context window for output
    out_units = total_context_window_k * 1000
    input_size = int(out_units * 0.95)
    return input_size

if __name__ == '__main__':
    # Get API keys
    from dotenv import load_dotenv

    load_dotenv()





    # Max tokens 16k
    # Input: $0.5/1M tokens
    # Output $1.5/1M tokens
    model1 = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
    gpt3_outputs = query_a_model(model1, os.path.join('example_inputs', '4187756.txt'), get_input_size_limit(16))

    # Max tokens 128k
    # Input: $5.00 /1M tokens
    # Output $15.00 /1M tokens
    # model2 = ChatOpenAI(model="gpt-4o", temperature=0)
    # gpt4_outputs = query_a_model(model2, os.path.join('example_inputs', '4187756.txt'), get_input_size_limit(128))

    # TODO: fix auth for this

    # Max tokens 128k
    # Input: $0.00125 / 1k characters
    # Output $0.0025 / 1k characters
    # import vertexai
    #
    # PROJECT_ID = "[medplantmining]"  # @param {type:"string"}
    # REGION = "europe-west2"  # @param {type:"string"}
    #
    # # Initialize Vertex AI SDK
    # vertexai.init(project=PROJECT_ID, location=REGION)
    # model3 = ChatVertexAI(model="gemini-pro", temperature=0)
    # gemini_outputs = query_a_model(model3, os.path.join('example_inputs', '4187756.txt'), get_input_size_limit(128))

    # Max tokens 200k
    # Input: $3 / MTok
    # Output $15 / MTok
    # model4 = ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=0)
    # claude_outputs = query_a_model(model4, os.path.join('example_inputs', '4187756.txt'), get_input_size_limit(200))

    # Max tokens 32k
    # Input: $4/1M tokens
    # Output $8.1/1M tokens
    # model5 = ChatMistralAI(model="mistral-large-latest", temperature=0)
    # mistral_outputs = query_a_model(model5, os.path.join('example_inputs', '4187756.txt'),, get_input_size_limit(32))
    pass
