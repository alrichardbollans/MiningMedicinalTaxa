import os
import pickle
import sys

import pandas as pd
from langchain_openai import ChatOpenAI

from rag_models.evaluating import NER_evaluation, precise_match, approximate_match, RE_evaluation, check_errors
from rag_models.running_models import query_a_model, get_input_size_limit, setup_models
from rag_models.structured_output_schema import get_all_human_annotations_for_corpus_id, annotation_info


def _get_chunks_to_tweak_with():
    ''' Splits into train/test chunks but will be easiest to use whole papers instead.'''
    from sklearn.model_selection import train_test_split

    train, test = train_test_split(annotation_info, test_size=0.1, shuffle=True)

    test.to_csv(os.path.join('outputs', 'for_hparam_tuning.csv'), index=False)
    train.to_csv(os.path.join('outputs', 'for_testing.csv'), index=False)


def _get_train_test_papers():
    ids = annotation_info['corpus_id'].unique().tolist()
    # assert len(ids) == 10
    train = ['4187756']  # just use first paper
    test = [c for c in ids if c not in train]
    return train, test


def assessing_hparams():
    # Get API keys
    from dotenv import load_dotenv

    load_dotenv(os.path.join(repo_path, 'MedicinalPlantMining', 'rag_models', '.env'))

    for corpus_id in train:
        human_annotations = get_all_human_annotations_for_corpus_id(corpus_id, check=False)

        # model1 = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
        # gpt3_outputs = query_a_model(model1, os.path.join(base_text_path, f'{corpus_id}.txt'),
        #                              get_input_size_limit(16))
        # with open("hparam_outputs.pickle", "wb") as file_:
        #     pickle.dump(gpt3_outputs, file_)

        gpt3_outputs = pickle.load(open("hparam_outputs.pickle", "rb", -1))

        precise_gpt3_precision, precise_gpt3_recall, precise_gpt3_f1_score = NER_evaluation(gpt3_outputs, human_annotations, precise_match)
        approximate_gpt3_precision, approximate_gpt3_recall, approximate_gpt3_f1_score = NER_evaluation(gpt3_outputs, human_annotations,
                                                                                                        approximate_match)

        print(precise_gpt3_precision, precise_gpt3_recall, precise_gpt3_f1_score)
        precise_re_results = RE_evaluation(gpt3_outputs, human_annotations, precise_match)
        check_errors(gpt3_outputs, human_annotations)


def full_evaluation():
    for corpus_id in test:
        human_annotations = get_all_human_annotations_for_corpus_id(corpus_id)

        all_models = setup_models()

        for m in all_models:
            gpt3_outputs = query_a_model(m, os.path.join(base_text_path, f'{corpus_id}.txt'),
                                         get_input_size_limit(16))
            precise_gpt3_precision, precise_gpt3_recall, precise_gpt3_f1_score = NER_evaluation(gpt3_outputs, human_annotations, precise_match)
            approximate_gpt3_precision, approximate_gpt3_recall, approximate_gpt3_f1_score = NER_evaluation(gpt3_outputs, human_annotations,
                                                                                                            approximate_match)


def main():
    assessing_hparams()


if __name__ == '__main__':
    train, test = _get_train_test_papers()
    repo_path = os.environ.get('KEWSCRATCHPATH')
    base_text_path = os.path.join(repo_path, 'MedicinalPlantMining', 'annotated_data', 'top_10_medicinal_hits', 'text_files')
    main()
