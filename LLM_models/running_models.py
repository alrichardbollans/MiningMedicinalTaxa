import os
import pickle
import string

import langchain_core
import pydantic_core

from LLM_models.loading_files import read_file_and_chunk, split_text_chunks
from LLM_models.making_examples import example_messages
from LLM_models.rag_prompting import standard_medicinal_prompt
from LLM_models.structured_output_schema import deduplicate_and_standardise_output_taxa_lists, TaxaData


def sanitize_text(s: str):
    table = str.maketrans(dict.fromkeys(string.punctuation))
    out = s.translate(table)
    return out


def query_a_model(model, text_file: str, context_window: int, pkl_dump: str = None, single_chunk: bool = True, examples=example_messages) -> TaxaData:
    text_chunks = read_file_and_chunk(text_file, context_window)
    if single_chunk:
        # For most of analysis, will be testing on single chunks as this is how we've annotated them.
        # In this instance, the chunks should fit in the context window
        assert len(text_chunks) == 1
    # A few different methods, depending on the specific model are used to get a structured output
    # and this is handled by with_structured_output. See https://python.langchain.com/docs/how_to/structured_output/
    extractor = standard_medicinal_prompt | model.with_structured_output(schema=TaxaData, include_raw=False)
    try:

        extractions = extractor.batch(
            [{"text": text, "examples": examples} for text in text_chunks],
            {"max_concurrency": 1},  # limit the concurrency by passing max concurrency! Otherwise Requests rate limit exceeded
        )
    except (langchain_core.exceptions.OutputParserException, pydantic_core._pydantic_core.ValidationError) as e:
        # When there is too much info extracted the extractor can't parse the output json, so make chunks smaller.
        # This can also happen because of limits on model max output tokens
        print(f'Warning: reducing size of chunk as output json is too large to parse. For file {text_file}')

        new_chunks = split_text_chunks(text_chunks)
        print(f'Length of old chunk: {len(text_chunks[0])}')

        extractions = []
        for text in new_chunks:
            try:
                chunk_output = extractor.invoke({"text": text, "examples": examples})
                extractions.append(chunk_output)
            except Exception as e:
                more_chunks = split_text_chunks([text])
                for more_text in more_chunks:
                    try:
                        chunk_output = extractor.invoke({"text": more_text, "examples": examples})
                        extractions.append(chunk_output)
                    except Exception as e:
                        # print(f'Unknown error "{e}" for text with length {len(more_text)}: {more_text}')
                        even_more_chunks = split_text_chunks([more_text])
                        for even_more_text in even_more_chunks:
                            try:
                                chunk_output = extractor.invoke({"text": even_more_text, "examples": examples})
                                extractions.append(chunk_output)
                            except Exception as e:
                                print(f'Unknown error "{e}" for text with length {len(even_more_text)}: {even_more_text}')

    output = []

    for extraction in extractions:
        if extraction.taxa is not None:
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
    from langchain_openai import ChatOpenAI

    # Get API keys
    from dotenv import load_dotenv

    load_dotenv()
    out = {}
    # A selection of models that support .with_structured_output https://python.langchain.com/v0.3/docs/integrations/chat/
    # Try to use the best from each company, and use a specified stable version.
    # If any work particularly well then also test cheaper versions e.g. gpt-mini, claude haiku

    hparams = {'temperature': 0}

    # https://platform.openai.com/docs/models/gpt-4o
    # Max tokens 128k
    # Input: $5.00 /1M tokens
    # Output $15.00 /1M tokens
    model1 = ChatOpenAI(model="gpt-4o-2024-08-06", **hparams)
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
    # https://ai.google.dev/gemini-api/docs/models/gemini
    # Note gemini doesn't like nested pydantic models: https://github.com/langchain-ai/langchain-google/issues/659#issuecomment-2568319643
    # model2 = ChatVertexAI(model="gemini-1.5-pro-002", max_tokens=8192, **hparams)
    # out['gemini'] = [model2, get_input_size_limit(128)]

    # Max tokens 200k
    # Input: $3 / MTok
    # Output $15 / MTok
    # https://docs.anthropic.com/en/docs/about-claude/models
    model3 = ChatAnthropic(model="claude-3-5-sonnet-20241022", max_tokens=4096, **hparams)
    out['claude'] = [model3, get_input_size_limit(200)]

    # Max tokens 32k
    # Input: $2/1M tokens
    # Output $6/1M tokens
    # https://mistral.ai/technology/#pricing
    # model4 = ChatMistralAI(model="mistral-large-2407", **hparams)
    # out['mistral'] = [model4, get_input_size_limit(128)]

    # Llama api still experimental and token limit is too small via groq
    # model5 = ChatGroq(model="llama3-70b-8192",
    #                   **hparams)
    # out['llama'] = [model5, get_input_size_limit(8)]

    # Llama 3.1 405B Instruct
    # Max tokens 131k
    # Input/Output: $3/1M tokens

    from langchain_fireworks import ChatFireworks

    model5 = ChatFireworks(
        model="accounts/fireworks/models/llama-v3p1-405b-instruct", **hparams)
    out['llama'] = [model5, get_input_size_limit(131)]

    # DeepSeek V3
    # Created 30/12/2024
    # Max tokens 128k
    # Input/Output: $0.07/1.10/1M tokens

    from langchain_deepseek import ChatDeepSeek
    model6 = ChatDeepSeek(
        model="deepseek-chat", **hparams)
    out['deepseek'] = [model6, get_input_size_limit(128)]

    return out


if __name__ == '__main__':
    models = setup_models()

    example_model_name = 'deepseek'
    repo_path = os.environ.get('KEWSCRATCHPATH')
    base_text_path = os.path.join(repo_path, 'MedicinalPlantMining', 'annotated_data', 'top_10_medicinal_hits', 'text_files')
    base_chunk_path = os.path.join(repo_path, 'MedicinalPlantMining', 'annotated_data', 'top_10_medicinal_hits', 'chunks', 'selected_chunks')

    example_model_outputs = query_a_model(models[example_model_name][0], os.path.join(base_chunk_path, '4187756.txt_chunk_15.txt'),
                                          models[example_model_name][1],
                                          single_chunk=True)
    print(example_model_outputs)
