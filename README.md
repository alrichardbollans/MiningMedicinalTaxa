This is a collection of packages for downloading and filtering corpora (currently CORE v2022) and then exploring named entity recognition and relation extraction in the open access texts.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alrichardbollans/MiningMedicinalTaxa/blob/main/notebook/MiningMedicinalTaxaExample.ipynb)


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
import json
from SciBert.running_scibert import load_scibert, query_scibert
from LLM_models.evaluating import clean_model_annotations_using_taxonomy_knowledge

models = load_scibert()

model_outputs = query_scibert(models, 'test.txt',
                       json_dump='output_scibert.json',
                       run_re=True)
# If you want to clean outputs by removing annotations with unknown scientific names:
model_outputs = clean_model_annotations_using_taxonomy_knowledge(model_outputs)

with open('output_scibert_filtered.json', "w") as file_:
    json_out = model_outputs.model_dump(mode="json")
    json.dump(json_out, file_)
```

## R

The R wrapper at [`R/mining_medicinal_taxa.R`](R/mining_medicinal_taxa.R) uses [reticulate](https://rstudio.github.io/reticulate/) to call the Python package from R and returns results in JSON format.

First install the packages:

```r
install.packages("reticulate")
source("https://raw.githubusercontent.com/alrichardbollans/MiningMedicinalTaxa/main/R/mining_medicinal_taxa.R")
install_mining_medicinal_taxa()
# Restart R when this finishes
```
Usage example

```r
source("https://raw.githubusercontent.com/alrichardbollans/MiningMedicinalTaxa/main/R/mining_medicinal_taxa.R")
mmt_activate()

# Sample text (or use your own .txt)
sample_url <- "https://raw.githubusercontent.com/alrichardbollans/MiningMedicinalTaxa/main/R/sample.txt"
txt_file   <- "sample.txt"
download.file(sample_url, txt_file, mode = "wb")


# SciBERT
models <- load_scibert_models()      # ~400 MB on first run, cached after
scibert_results <- run_scibert(txt_file, models,
                               output_json = "scibert_output.json",
                               run_re = TRUE, # run_re = FALSE for NER only
                               clean_names = FALSE) # clean names = TRUE clean outputs by removing annotations with unknown scientific names
print_taxa(scibert_results, "SciBERT")

# GPT set your OpenAI API key first
Sys.setenv(OPENAI_API_KEY = "sk-...")
gpt_results <- run_gpt(txt_file,
                       output_json      = "gpt_output.json",
                       context_window_k = 5,
                       model = "gpt-4o-2024-08-06",
                       clean_names = FALSE) # clean names = TRUE clean outputs by removing annotations with unknown scientific names
print_taxa(gpt_results, "GPT-4o")
```


### Manual verification

Outputs from this process (the `json_dump` files) can be manually verified using our reference verifier shiny app, hosted
here: https://huggingface.co/spaces/alrichardbollans/MedicinalTaxonVerifier
