import json
import os
import subprocess
from collections import defaultdict

import numpy as np
import pandas as pd
import tiktoken
from langchain_openai import ChatOpenAI

from LLM_models.loading_files import read_file_and_chunk
from LLM_models.making_examples import example_messages
from LLM_models.rag_prompting import standard_medicinal_prompt
from LLM_models.running_models import get_input_size_limit
from LLM_models.structured_output_schema import get_chunk_filepath_from_chunk_id, TaxaData, get_all_human_annotations_for_chunk_id


def run_example():
    # Just an example to manually check outputs
    issues = [100]
    top_model = "claude-3-5-sonnet-20241022"
    cheaper_model = 'claude-3-haiku-20240307'
    model1 = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
    for id in issues:
        text_chunks = read_file_and_chunk(get_chunk_filepath_from_chunk_id(id), get_input_size_limit(16))
        extractor = standard_medicinal_prompt | model1.with_structured_output(schema=TaxaData, include_raw=True)
        chunk_output = extractor.invoke({"text": text_chunks[0], "examples": example_messages})
        print(chunk_output)
        relevant_example = chunk_output['raw'].additional_kwargs['tool_calls'][0]['function']['arguments']
        print(relevant_example)
        relevant_example = '{"taxa":[{"scientific_name":"Apium graveolens","medical_conditions":["rash"]},{"scientific_name":"Solandra spp.","medical_conditions":["vomiting","diarrhea","pupils dilate"]},{"scientific_name":"Prunus spp.","medical_conditions":["twitching","difficult breathing","coma"]},{"scientific_name":"Helleborus niger","medical_conditions":["upset stomach","purging","numbing of mouth"]},{"scientific_name":"Codiaeum spp.","medical_conditions":["rash","irritation of mouth and throat"]},{"scientific_name":"Cyclamen spp.","medical_conditions":["rash"]},{"scientific_name":"Narcissus pseudonarcissus","medical_conditions":["vomiting","diarrhea","trembling","convulsions"]},{"scientific_name":"Daphne mezereum","medical_conditions":["vomiting","diarrhea","stupor","convulsions"]},{"scientific_name":"Dieffenbachia spp.","medical_conditions":["irritation of mouth and throat","voice loss"]},{"scientific_name":"Sambucus spp.","medical_conditions":["nausea","digestive upset"]},{"scientific_name":"Colocasia spp.","medical_conditions":["irritation of mouth and throat"]},{"scientific_name":"Hedera helix","medical_conditions":["excitement","difficult breathing","coma"]},{"scientific_name":"Ficus spp.","medical_conditions":["rash"]},{"scientific_name":"Digitalis purpurea","medical_conditions":["irregular heartbeat and pulse","digestive upset"]},{"scientific_name":"Mirabilis jalapa","medical_conditions":["vomiting","diarrhea","stomach pain"]},{"scientific_name":"Laburnum anagyroides","medical_conditions":["incoordination","vomiting","convulsions","coma"]},{"scientific_name":"Ilex spp.","medical_conditions":["vomiting","diarrhea","stupor"]},{"scientific_name":"Hyacinthus orientalis","medical_conditions":["intense indigestion"]},{"scientific_name":"Hydrangea spp.","medical_conditions":["nausea","vomiting","diarrhea"]},{"scientific_name":"Iris spp.","medical_conditions":["rash","severe digestive upset","purging"]},{"scientific_name":"Delphinium spp.","medical_conditions":["digestive upset","excitement/depression"]},{"scientific_name":"Eriobotrya japonica","medical_conditions":["vomiting","labored breathing","convulsions"]},{"scientific_name":"Convallaria majalis","medical_conditions":["heart stimulant","dizziness","vomiting"]},{"scientific_name":"Phoradendron spp.","medical_conditions":["severe indigestion","cardiovascular collapse"]},{"scientific_name":"Aconitum spp.","medical_conditions":["tingling lips/tongue","slowing heart rate"]},{"scientific_name":"Ipomoea violacea","medical_conditions":["nausea","euphoria","hallucinations"]},{"scientific_name":"Solanum spp.","medical_conditions":["nausea","dizziness","pupils dilate"]},{"scientific_name":"Quercus spp.","medical_conditions":["constipation","bloody stools","kidney damage"]},{"scientific_name":"Nerium oleander","medical_conditions":["nausea","irregular pulse","paralysis"]},{"scientific_name":"Pastinaca sativa","medical_conditions":["rash"]},{"scientific_name":"Philodendron spp.","medical_conditions":["irritation of mouth and throat"]},{"scientific_name":"Pieris japonica","medical_conditions":["vomiting","low blood pressure","convulsions"]},{"scientific_name":"Euphorbia pulcherrima","medical_conditions":["rash","vomiting","abdominal pain","diarrhea"]},{"scientific_name":"Papaver spp.","medical_conditions":["stupor","coma","slow breathing"]},{"scientific_name":"Solanum tuberosum","medical_conditions":["vomiting","diarrhea","shock","paralysis"]},{"scientific_name":"Primula obconica","medical_conditions":["rash"]},{"scientific_name":"Ligustrum vulgare","medical_conditions":["upset stomach","pain","vomiting","diarrhea"]},{"scientific_name":"Lantana camara","medical_conditions":["intestinal upset","muscular weakness"]},{"scientific_name":"Rhododendron spp.","medical_conditions":["vomiting","low blood pressure","convulsions"]},{"scientific_name":"Rheum rhaponticum","medical_conditions":["severe abdominal pain","vomiting","weakness"]},{"scientific_name":"Euphorbia spp.","medical_conditions":["rash"]},{"scientific_name":"Lathyrus odoratus","medical_conditions":["paralysis (when eaten in large quantity)"]},{"scientific_name":"Lycopersicon esculentum","medical_conditions":["vomiting","diarrhea","shock","paralysis"]},{"scientific_name":"Tulipa spp.","medical_conditions":["vomiting","diarrhea","stomach pain"]},{"scientific_name":"Wisteria spp.","medical_conditions":["vomiting","diarrhea","abdominal pain"]},{"scientific_name":"Thevetia peruviana","medical_conditions":["vomiting","diarrhea","abdominal pain","headache"]},{"scientific_name":"Taxus spp.","medical_conditions":["vomiting","diarrhea","circulatory collapse"]}]}'


def get_finetuning_data():
    system_content = standard_medicinal_prompt.messages[0].prompt.template
    output_data = []
    for chunk_id in df_for_hparam_tuning['id'].unique().tolist():
        text_chunks = read_file_and_chunk(get_chunk_filepath_from_chunk_id(chunk_id), training_max_content_length)

        assert len(text_chunks) == 1
        human_content = text_chunks[0]

        human_annotations = get_all_human_annotations_for_chunk_id(chunk_id, check=False, standardise_annotations=False)
        assistant_output = human_annotations.dict()
        # null=None
        # for t in assistant_output['taxa']:
        #     if t['medicinal_effects'] is None:
        #         t['medicinal_effects']=''
        #     if t['medical_conditions'] is None:
        #         t['medical_conditions']=
        string_assistant_answer = str(assistant_output)
        dict_for_chunk = {"messages": [{'role': 'system', 'content': system_content},
                                       {'role': 'user', 'content': human_content},
                                       {'role': 'assistant', 'content': string_assistant_answer}]}
        output_data.append(dict_for_chunk)
    with open(fine_tuning_data_file, 'w') as f:
        for d in output_data:
            json.dump(d, f)
            f.write('\n')


def check_data_format_errors(dataset):
    # From https://cookbook.openai.com/examples/chat_finetuning_data_prep
    # Format error checks
    format_errors = defaultdict(int)

    for ex in dataset:
        if not isinstance(ex, dict):
            format_errors["data_type"] += 1
            continue

        messages = ex.get("messages", None)
        if not messages:
            format_errors["missing_messages_list"] += 1
            continue

        for message in messages:
            if "role" not in message or "content" not in message:
                format_errors["message_missing_key"] += 1

            if any(k not in ("role", "content", "name", "function_call", "weight") for k in message):
                format_errors["message_unrecognized_key"] += 1

            if message.get("role", None) not in ("system", "user", "assistant", "function"):
                format_errors["unrecognized_role"] += 1

            content = message.get("content", None)
            function_call = message.get("function_call", None)

            if (not content and not function_call) or not isinstance(content, str):
                format_errors["missing_content"] += 1

        if not any(message.get("role", None) == "assistant" for message in messages):
            format_errors["example_missing_assistant_message"] += 1

    if format_errors:
        print("Found errors:")
        for k, v in format_errors.items():
            print(f"{k}: {v}")
        raise ValueError
    else:
        print("No errors found")

    # Warnings and tokens counts

    encoding = tiktoken.get_encoding("cl100k_base")

    # not exact!
    # simplified from https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    def num_tokens_from_messages(messages, tokens_per_message=3, tokens_per_name=1):
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3
        return num_tokens

    def num_assistant_tokens_from_messages(messages):
        num_tokens = 0
        for message in messages:
            if message["role"] == "assistant":
                num_tokens += len(encoding.encode(message["content"]))
        return num_tokens

    def print_distribution(values, name):
        print(f"\n#### Distribution of {name}:")
        print(f"min / max: {min(values)}, {max(values)}")
        print(f"mean / median: {np.mean(values)}, {np.median(values)}")
        print(f"p5 / p95: {np.quantile(values, 0.1)}, {np.quantile(values, 0.9)}")

    n_missing_system = 0
    n_missing_user = 0
    n_messages = []
    convo_lens = []
    assistant_message_lens = []

    for ex in dataset:
        messages = ex["messages"]
        if not any(message["role"] == "system" for message in messages):
            n_missing_system += 1
        if not any(message["role"] == "user" for message in messages):
            n_missing_user += 1
        n_messages.append(len(messages))
        convo_lens.append(num_tokens_from_messages(messages))
        assistant_message_lens.append(num_assistant_tokens_from_messages(messages))

    print("Num examples missing system message:", n_missing_system)
    print("Num examples missing user message:", n_missing_user)
    print_distribution(n_messages, "num_messages_per_example")
    print_distribution(convo_lens, "num_total_tokens_per_example")
    print_distribution(assistant_message_lens, "num_assistant_tokens_per_example")
    n_too_long = sum(l > training_max_content_length for l in convo_lens)
    print(f"\n{n_too_long} examples may be over the 16,385 token limit, they will be truncated during fine-tuning")

    # def check_data_pricing(dataset):
    # Pricing and default n_epochs estimate
    MAX_TOKENS_PER_EXAMPLE = 16385

    TARGET_EPOCHS = 3
    MIN_TARGET_EXAMPLES = 100
    MAX_TARGET_EXAMPLES = 25000
    MIN_DEFAULT_EPOCHS = 1
    MAX_DEFAULT_EPOCHS = 25

    n_epochs = TARGET_EPOCHS
    n_train_examples = len(dataset)
    if n_train_examples * TARGET_EPOCHS < MIN_TARGET_EXAMPLES:
        n_epochs = min(MAX_DEFAULT_EPOCHS, MIN_TARGET_EXAMPLES // n_train_examples)
    elif n_train_examples * TARGET_EPOCHS > MAX_TARGET_EXAMPLES:
        n_epochs = max(MIN_DEFAULT_EPOCHS, MAX_TARGET_EXAMPLES // n_train_examples)

    n_billing_tokens_in_dataset = sum(min(MAX_TOKENS_PER_EXAMPLE, length) for length in convo_lens)
    print(f"Dataset has ~{n_billing_tokens_in_dataset} tokens that will be charged for during training")
    print(f"By default, you'll train for {n_epochs} epochs on this dataset")
    print(f"By default, you'll be charged for ~{n_epochs * n_billing_tokens_in_dataset} tokens")
    print('(base training cost per 1M input tokens ÷ 1M) × number of tokens in the input file × number of epochs trained')


def finetune_model():
    # Following https://platform.openai.com/docs/guides/fine-tuning/#create-a-fine-tuned-model
    # $25.000 / 1M training tokens
    from openai import OpenAI
    client = OpenAI()

    file = client.files.create(
        file=open(fine_tuning_data_file, "rb"),
        purpose="fine-tune"
    )

    client.fine_tuning.jobs.create(
        training_file=file.id,
        model="gpt-4o-2024-08-06"
    )


def check_jobs():
    from openai import OpenAI
    client = OpenAI()

    # List 10 fine-tuning jobs
    print(client.fine_tuning.jobs.list(limit=10))
    job_id = "ftjob-5LFmDazZTCselWl0K96Ofk0z"
    api_key = os.getenv('OPENAI_API_KEY')
    return_code = subprocess.call(f'curl https://api.openai.com/v1/fine_tuning/jobs/{job_id} -H "Authorization: Bearer {api_key}"', shell=True)

    result_files = [
        "file-DzCssK7eaZ8Lp6Q7ZzYeVt"
    ]
    import base64
    counter = 0
    for r in result_files:
        counter += 1
        content = client.files.content(r)
        ans = base64.b64decode(content.content).decode()
        with open(f'inputs/fine_tuning_results{str(counter)}.csv', 'w') as f:
            f.write(ans)

        # subprocess.call(f'curl https://api.openai.com/v1/files/{f}/content -H "Authorization: Bearer {api_key}" > inputs/results{str(counter)}.csv', shell=True)
    pass


def main():
    # run_example()
    # get_finetuning_data()
    # # Load the dataset
    # with open(fine_tuning_data_file, 'r', encoding='utf-8') as f:
    #     dataset = [json.loads(line) for line in f]
    #     check_data_format_errors(dataset)
    #
    # finetune_model()
    check_jobs()


if __name__ == '__main__':
    from dotenv import load_dotenv

    load_dotenv()
    df_for_hparam_tuning = pd.read_csv(os.path.join('evaluating', 'outputs', 'for_hparam_tuning.csv'))
    fine_tuning_data_file = os.path.join('inputs', 'myfinetuning_data.jsonl')
    training_max_content_length = 65536

    main()
