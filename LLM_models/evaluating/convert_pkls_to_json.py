import json
import os
import pickle

from LLM_models.checking_and_summarising_annotations import get_chunk_filepath_from_chunk_id
from LLM_models.loading_files import get_txt_from_file
from LLM_models.structured_output_schema import TaxaData


def main():
    in_dir_name = 'outputs/model_pkls'
    out_dir_name = 'outputs/model_jsons'
    for file_name in os.listdir(in_dir_name):
        if file_name.endswith(".pickle"):
            json_file_name = file_name.replace('.pickle', '.json')

            # Load previous pkl
            with open(f'{in_dir_name}/{file_name}', "rb") as file_:
                output = pickle.load(file_)

            # write it as json
            with open(f'{out_dir_name}/{json_file_name}', "w") as file_:
                json_out = output.model_dump(mode="json")
                json.dump(json_out, file_)

            # read it back and add text
            c_id = int(file_name.split('_')[0])
            text_file = get_chunk_filepath_from_chunk_id(c_id)
            text = get_txt_from_file(text_file)
            with open(f'{out_dir_name}/{json_file_name}', "r") as file_:
                json_dict = json.load(file_)
                read_output = TaxaData.model_validate(json_dict)
                read_output.text = text

            ## then write it again
            with open(f'{out_dir_name}/{json_file_name}', "w") as file_:
                json_out = read_output.model_dump(mode="json")
                json.dump(json_out, file_)
            fields_ = ['scientific_name', 'medical_conditions', 'medicinal_effects']

            if hasattr(read_output, 'text'):
                assert read_output.text == text
            if len(read_output.taxa) > 0:
                for i in range(len(read_output.taxa)):
                    read_taxon = read_output.taxa[i]
                    pkled_taxon = output.taxa[i]
                    for field in fields_:
                        read_field = getattr(read_taxon, field)
                        pkled_field = getattr(pkled_taxon, field)
                        if read_field is not None or pkled_field is not None:
                            if read_field == read_field and pkled_field == pkled_field:
                                assert read_field == pkled_field


if __name__ == '__main__':
    main()
