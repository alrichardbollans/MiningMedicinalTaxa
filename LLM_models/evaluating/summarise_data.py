import os

import pandas as pd

from LLM_models.structured_output_schema import summarise_annotations, valid_chunk_annotation_info, annotation_info


def main():
    for_hparam_tuning = pd.read_csv(os.path.join('outputs', 'for_hparam_tuning.csv'))
    for_testing = pd.read_csv(os.path.join('outputs', 'for_testing.csv'))
    summarise_annotations(for_hparam_tuning['id'].unique().tolist(), os.path.join('outputs', 'tuning_data_summary.csv'))
    summarise_annotations(for_testing['id'].unique().tolist(), os.path.join('outputs', 'testing_data_summary.csv'))

    assert valid_chunk_annotation_info['reference_only'].unique().tolist() == ['no']
    summarise_annotations(valid_chunk_annotation_info['id'].unique().tolist(), os.path.join('outputs', 'valid_data_summary.csv'))
    summarise_annotations(annotation_info['id'].unique().tolist(), os.path.join('outputs', 'all_data_summary.csv'))


if __name__ == '__main__':
    main()
