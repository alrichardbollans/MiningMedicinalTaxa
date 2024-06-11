import json
import os
import re

from google.generativeai.types import GenerateContentResponse

from chunk_level_models import get_full_prompt_for_txt_file
from chunk_level_models.evaluation import chunkwise_evaluation, precise_output_annotation_match, approximate_output_annotation_match
from testing.evaluation_methods import read_annotation_json


def get_gemini_response(model_name: str, prompt: str):
    import google.generativeai as genai
    import os
    from dotenv import load_dotenv

    load_dotenv()

    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    genai.configure(api_key=GOOGLE_API_KEY)

    # Set up the model
    generation_config = {
        "temperature": 0
        # "top_p": 0.95,
        # "top_k": 64,
    }

    model = genai.GenerativeModel(model_name=model_name, generation_config=generation_config)

    response = model.generate_content(prompt)

    print(response.text)
    return response


def validate_json(response_json):
    pass


def get_response_json(response_text: str):
    result = re.search('```json(.*)```', response_text)
    json_string = result.group(1)
    response_json = json.loads(json_string)

    validate_json(response_json)
    return response_json


if __name__ == '__main__':
    # FOllowing
    # https://ai.google.dev/gemini-api/docs/get-started/tutorial?lang=python
    # https://cloud.google.com/vertex-ai/generative-ai/docs/samples/generativeaionvertexai-gemini-pdf#generativeaionvertexai_gemini_pdf-python

    # Playing here:
    # https://console.cloud.google.com/vertex-ai/generative/language/create/text?hl=en&project=medplantmining

    # prompt = get_full_prompt_for_txt_file(os.path.join('example_inputs', '4187756.txt'))
    #
    # response = get_gemini_response("gemini-1.5-pro-001", prompt)
    response_text = '```json{"annotations": [{"entity": "hypericum perforatum", "Medical condition": ["depression", "burns", "snakebites", "anxiety", "nervous unrest", "psychovegetative disturbances", "depressive moods", "nervous exhaustion", "nerve damage", "long-term emotional stress"], "Medicinal effect": ["diuretic", "anti-depressant", "anxiolytic", "reduce alcohol intake", "appeases nicotine withdrawal"]}, {"entity": "allium sativum", "Medical condition": [], "Medicinal effect": []}, {"entity": "ginkgo biloba", "Medical condition": [], "Medicinal effect": []}, {"entity": "johanniskraut", "Medical condition": ["psychovegetative disturbances", "depressive moods", "anxiety", "nervous unrest"], "Medicinal effect": ["mild antidepressant action"]}, {"entity": "ephedra", "Medical condition": [], "Medicinal effect": ["raising blood pressure", "stressing the body’s circulatory system"]}]}```'

    model_annotations = get_response_json(response_text)

    # TODO: need to include ner_annotations better in evaluation
    ner_annotations, human_re_annotations = read_annotation_json('../testing/test_medicinal_01/tasks_completed', '4187756', '0')
    precision, recall, f1_score = chunkwise_evaluation(model_annotations, human_re_annotations, precise_output_annotation_match)
    print(f'precision: {precision}, recall: {recall}, f1_score: {f1_score}')
    precision, recall, f1_score = chunkwise_evaluation(model_annotations, human_re_annotations, approximate_output_annotation_match)
    print(f'precision: {precision}, recall: {recall}, f1_score: {f1_score}')
