import json
import os
import pathlib

from LLM_models.evaluating import clean_model_annotations_using_taxonomy_knowledge
from LLM_models.evaluating.fine_tuned_model import get_fine_tuned_model
from LLM_models.running_models import query_a_model


def run_model():
    ft_model = get_fine_tuned_model()['gpt4o_FT']
    in_dir = 'chunks'
    out_dir = 'extractions'
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)

    for file_name in os.listdir(in_dir):
        print(f"Processing {file_name}")
        model_outputs = query_a_model(ft_model[0], os.path.join(in_dir, file_name),
                                      ft_model[1])
        #
        # # If you want to clean outputs by removing annotations with unknown scientific names:
        clean_model_outputs = clean_model_annotations_using_taxonomy_knowledge(model_outputs)

        with open(os.path.join(out_dir, f'{file_name.replace('.txt', '.json')}'), "w") as file_:
            json_out = clean_model_outputs.model_dump(mode="json")
            json.dump(json_out, file_)


def main():
    run_model()


if __name__ == '__main__':
    main()
