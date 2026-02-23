## Usage Example

A high-level example of how to run a model. Of course, you could reimplement this with the langchain functions using the prompt we have generated and
`TaxaData` output class for finer control.

```python
import json
from LLM_models.running_models import query_a_model, setup_models
from LLM_models.evaluating import clean_model_annotations_using_taxonomy_knowledge

# Specify a .env file containing your deepseek API key in the form DEEPSEEK_API_KEY="key",
# or alternatively specify the apikey directly with the apikey parameter
models = setup_models(dotenv_path='.env')


# models are output as a dictionary indexed by model names.
# Each dictionary entry is a list containing [MODEL, CONTEXT_WINDOW]
# Using 'deepseek' in this example
model_outputs = query_a_model(models['deepseek'][0], 'path_to_input_txt_file.txt',
              models['deepseek'][1],json_dump='output_json_path.json')

# If you want to clean outputs by removing annotations with unknown scientific names:
model_outputs = clean_model_annotations_using_taxonomy_knowledge(model_outputs)

with open('output_json_path_filtered.json', "w") as file_:
    json_out = model_outputs.model_dump(mode="json")
    json.dump(json_out, file_)

```

### Manual verification

Outputs from this process (the `json_dump` files) can be manually verified using our reference verifier shiny app, hosted
here: https://huggingface.co/spaces/alrichardbollans/MedicinalTaxonVerifier
