## Usage Example

A high-level example of how to run a model. Of course, you could reimplement this with the langchain functions using the prompt we have generated and
`TaxaData` output class for finer control.

```python
from LLM_models.running_models import query_a_model, setup_models

# Specify a .env file containing your deepseek API key in the form DEEPSEEK_API_KEY="key",
# or alternatively specify the apikey directly with the apikey parameter
models = setup_models(dotenv_path='.env')


# models are output as a dictionary indexed by model names.
# Each dictionary entry is a list containing [MODEL, CONTEXT_WINDOW]
# Using 'deepseek' in this example
query_a_model(models['deepseek'][0], 'path_to_input_txt_file.txt',
              models['deepseek'][1],json_dump='output_json_path.json')

```

### Manual verification

Outputs from this process (the `json_dump` files) can be manually verified using our reference verifier shiny app, hosted
here: https://huggingface.co/spaces/alrichardbollans/MedicinalTaxonVerifier
