import os
import pickle

import langchain_core

from rag_models.loading_files import read_file_and_chunk
from rag_models.rag_prompting import standard_medicinal_prompt
from rag_models.structured_output_schema import deduplicate_and_standardise_output_taxa_lists, TaxaData


def query_a_model(model, text_file: str, context_window: int, pkl_dump: str = None, single_chunk: bool = True) -> TaxaData:
    text_chunks = read_file_and_chunk(text_file, context_window)
    if single_chunk:
        # For most of analysis, will be testing on single chunks as this is how we've annotated them.
        # In this instance, the chunks should fit in the context window
        assert len(text_chunks) == 1
    # A few different methods, depending on the specific model are used to get a structured output
    # and this is handled by with_structured_output. See https://python.langchain.com/v0.1/docs/modules/model_io/chat/structured_output/
    extractor = standard_medicinal_prompt | model.with_structured_output(schema=TaxaData, include_raw=False)
    try:
        extractions = extractor.batch(
            [{"text": text} for text in text_chunks],
            {"max_concurrency": 1},  # limit the concurrency by passing max concurrency! Otherwise Requests rate limit exceeded
        )
    except langchain_core.exceptions.OutputParserException as e:
        print(e)
        # TODO: Resolve
        raise ValueError(f'resovlve this. Think it happens because json is too big. Resolve by making chunks smaller (less info)')
    # output = extractor.invoke({'text': text})
    output = []

    for extraction in extractions:
        output.extend(extraction.taxa)

    deduplicated_extractions = deduplicate_and_standardise_output_taxa_lists(output)

    if pkl_dump:
        with open(pkl_dump, "wb") as file_:
            pickle.dump(deduplicated_extractions, file_)

    return deduplicated_extractions


def get_input_size_limit(total_context_window_k: int):
    # Output tokens so far is a tiny fraction, so allow 5% of context window for output
    out_units = total_context_window_k * 1000
    input_size = int(out_units * 0.95)
    return input_size


def setup_models():
    from langchain_anthropic import ChatAnthropic
    from langchain_google_vertexai import ChatVertexAI
    from langchain_mistralai import ChatMistralAI
    from langchain_openai import ChatOpenAI
    from langchain_groq import ChatGroq

    # Get API keys
    from dotenv import load_dotenv

    load_dotenv()
    out = {}
    # A selection of models that support .with_structured_output https://python.langchain.com/v0.2/docs/integrations/chat/
    # Try to use the best from each company
    # If any work particularly well then also test cheaper versions e.g. gpt-mini, claude haiku

    # Max tokens 128k
    # Input: $5.00 /1M tokens
    # Output $15.00 /1M tokens
    model1 = ChatOpenAI(model="gpt-4o", temperature=0)
    out['gpt4o'] = [model1, get_input_size_limit(128)]

    # Auth seems to work now
    # installed gcloud following https://cloud.google.com/sdk/docs/install#deb
    # then need to follow https://cloud.google.com/docs/authentication/provide-credentials-adc#local-dev

    # Max tokens 128k
    # Input: $0.00125 / 1k characters
    # Output $0.0025 / 1k characters
    # import vertexai
    # PROJECT_ID = "[medplantmining]"  # @param {type:"string"}
    # REGION = "europe-west2"  # @param {type:"string"}
    # # # Initialize Vertex AI SDK
    # vertexai.init(project=PROJECT_ID, location=REGION)
    model2 = ChatVertexAI(model="gemini-1.5-pro-001", temperature=0)
    out['gemini'] = [model2, get_input_size_limit(128)]

    # Max tokens 200k
    # Input: $3 / MTok
    # Output $15 / MTok
    model3 = ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=0)
    out['gpt4o'] = [model3, get_input_size_limit(200)]

    # Max tokens 32k
    # Input: $4/1M tokens
    # Output $8.1/1M tokens
    model4 = ChatMistralAI(model="mistral-large-latest", temperature=0)
    out['mistral'] = [model4, get_input_size_limit(32)]

    # TODO: Llama api still experimental and token limit is too small via groq
    # model5 = ChatGroq(model="llama3-70b-8192",
    #                   temperature=0)
    # out['llama'] = [model5, get_input_size_limit(8)]

    return out


if __name__ == '__main__':
    models = setup_models()

    example_model_name = 'gemini'
    repo_path = os.environ.get('KEWSCRATCHPATH')
    base_text_path = os.path.join(repo_path, 'MedicinalPlantMining', 'annotated_data', 'top_10_medicinal_hits', 'text_files')

    example_model_outputs = query_a_model(models[example_model_name][0], os.path.join(base_text_path, '4187756.txt'), models[example_model_name][1])
    print(example_model_outputs)
