import os

from langchain_anthropic import ChatAnthropic
from langchain_mistralai import ChatMistralAI
from langchain_openai import ChatOpenAI
from langchain_google_vertexai import ChatVertexAI

from rag_models.rag_prompting import standard_medicinal_prompt
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


def setup_models():
    # Get API keys
    from dotenv import load_dotenv

    load_dotenv()
    out = {}

    # Max tokens 16k
    # Input: $0.5/1M tokens
    # Output $1.5/1M tokens
    # A model to play with on one annotated paper
    hparam_tuning_model = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
    out['gpt3.5'] = [hparam_tuning_model, get_input_size_limit(16)]

    # Max tokens 128k
    # Input: $5.00 /1M tokens
    # Output $15.00 /1M tokens
    model2 = ChatOpenAI(model="gpt-4o", temperature=0)
    out['gpt4o'] = [model2, get_input_size_limit(128)]

    # TODO: fix auth for this

    # Max tokens 128k
    # Input: $0.00125 / 1k characters
    # Output $0.0025 / 1k characters
    import vertexai
    PROJECT_ID = "[medplantmining]"  # @param {type:"string"}
    REGION = "europe-west2"  # @param {type:"string"}
    # # Initialize Vertex AI SDK
    vertexai.init(project=PROJECT_ID, location=REGION)
    model3 = ChatVertexAI(model="gemini-pro", temperature=0)
    out['gemini'] = [model3, get_input_size_limit(128)]


    # Max tokens 200k
    # Input: $3 / MTok
    # Output $15 / MTok
    model4 = ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=0)
    out['gpt4o'] = [model4, get_input_size_limit(200)]

    # Max tokens 32k
    # Input: $4/1M tokens
    # Output $8.1/1M tokens
    model5 = ChatMistralAI(model="mistral-large-latest", temperature=0)
    out['mistral'] = [model5, get_input_size_limit(32)]
    return out


if __name__ == '__main__':
    models = setup_models()

    example_model_name = 'gemini'
    example_model_outputs = query_a_model(models[example_model_name][0], os.path.join('example_inputs', '4187756.txt'), models[example_model_name][1])
    print(example_model_outputs)
