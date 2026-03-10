This is a collection of packages for downloading and filtering corpora (currently CORE v2022) and then exploring named entity recognition and relation extraction in the open access texts.

## LLM Usage Example

An example of how to run a model on a given text file using the prompt in `[prompting.py](LLM_models/prompting.py)` to extract data using the 
`TaxaData` structure.

First either clone the repo, or install with:

`pip install git+https://github.com/alrichardbollans/MiningMedicinalTaxa.git`

```python
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from LLM_models.running_models import query_a_model, get_input_size_limit
from LLM_models.evaluating import clean_model_annotations_using_taxonomy_knowledge

# Specify a .env file containing your deepseek API key in the form OPENAI_API_KEY="key",
# or alternatively specify the apikey directly with the apikey parameter
load_dotenv(dotenv_path='.env')

# Set up a GPT model (alternatively you can use our setup_models helper function to load all the LLMs)
gpt_model = ChatOpenAI(model="gpt-4o-2024-08-06", temperature=0)
context_window = get_input_size_limit(128)  # based on 128k max tokens

model_outputs = query_a_model(gpt_model, 'path_to_input_txt_file.txt',
                              context_window, json_dump='output_json_path.json')

# If you want to clean outputs by removing annotations with unknown scientific names:
model_outputs = clean_model_annotations_using_taxonomy_knowledge(model_outputs)

with open('output_json_path_filtered.json', "w") as file_:
    json_out = model_outputs.model_dump(mode="json")
    json.dump(json_out, file_)

```
## SciBERT Usage Example
An example of how to run the fine-tuned SciBERT NER + RE models to extract data from a text file. Set `run_re=False` for NER only.

```python
from SciBert.running_scibert import load_scibert, query_scibert
from LLM_models.evaluating import clean_model_annotations_using_taxonomy_knowledge

models = load_scibert()

model_output = query_scibert(models, 'test.txt',
                       json_dump='output_scibert.json',
                       run_re=True)
# If you want to clean outputs by removing annotations with unknown scientific names:
model_outputs = clean_model_annotations_using_taxonomy_knowledge(model_outputs)

with open('output_scibert_filtered.json', "w") as file_:
    json_out = model_outputs.model_dump(mode="json")
    json.dump(json_out, file_)
```

### Manual verification

Outputs from this process (the `json_dump` files) can be manually verified using our reference verifier shiny app, hosted
here: https://huggingface.co/spaces/alrichardbollans/MedicinalTaxonVerifier
