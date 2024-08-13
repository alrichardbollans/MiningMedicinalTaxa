import os
import pickle
import time

import pandas as pd
from langchain_openai import ChatOpenAI

from pre_labelling.evaluating import get_metrics_from_tp_fp_fn
from rag_models.evaluating import NER_evaluation, precise_match, approximate_match, RE_evaluation, check_errors, abbreviated_approximate_match, \
    abbreviated_precise_match
from rag_models.running_models import query_a_model, get_input_size_limit, setup_models
from rag_models.structured_output_schema import get_all_human_annotations_for_corpus_id, annotation_info, get_all_human_annotations_for_chunk_id


def _get_chunks_to_tweak_with():
    raise ValueError('Are you sure you want to redefine?')
    ''' Splits into train/test chunks but will be easiest to use whole papers instead.'''
    from sklearn.model_selection import train_test_split

    train, test = train_test_split(annotation_info, test_size=0.1, shuffle=True)

    test.to_csv(os.path.join('outputs', 'for_hparam_tuning.csv'), index=False)
    train.to_csv(os.path.join('outputs', 'for_testing.csv'), index=False)


def _get_train_test_papers():
    raise ValueError('Are you sure you want to redefine?')

    # Not going to use this now
    ids = annotation_info['corpus_id'].unique().tolist()
    # assert len(ids) == 10
    train = ['4187756']  # just use first paper
    test = [c for c in ids if c not in train]
    return train, test


def get_chunk_filepath_from_chunk_id(chunk_id: int):
    name = annotation_info[annotation_info['id'] == chunk_id]['name'].iloc[0]
    name = name.removeprefix('task_for_labelstudio_')
    idx = name.index('_chunk')
    name = name[:idx] + '.txt' + name[idx:]
    name = name.replace('.json', '.txt')
    return os.path.join(base_chunk_path, name)


def assess_model_on_chunk_list(chunk_list, model, context_window, out_dir, rerun: bool = True):
    """
    Given a list of chunk ids, runs the model on each chunk and collects outputs
    :param chunk_list:
    :param model:
    :param context_window:
    :param rerun:
    :return:
    """
    if rerun:
        time.sleep(1)
    ### NER
    (all_precise_NER_true_positives_in_ground_truths, all_precise_NER_true_positives_in_model_annotations, all_precise_NER_false_positives,
     all_precise_NER_false_negatives) = [], [], [], []
    (all_approx_NER_true_positives_in_ground_truths, all_approx_NER_true_positives_in_model_annotations, all_approx_NER_false_positives,
     all_approx_NER_false_negatives) = [], [], [], []

    ### Medical Conditions
    (all_preciseMedCondtrue_positives_in_ground_truths, all_preciseMedCondtrue_positives_in_model_annotations, all_preciseMedCondfalse_positives,
     all_preciseMedCondfalse_negatives) = [], [], [], []
    (all_approxMedCondtrue_positives_in_ground_truths, all_approxMedCondtrue_positives_in_model_annotations, all_approxMedCondfalse_positives,
     all_approxMedCondfalse_negatives) = [], [], [], []

    ## Medicinal Effects
    (all_preciseMedEfftrue_positives_in_ground_truths, all_preciseMedEfftrue_positives_in_model_annotations, all_preciseMedEfffalse_positives,
     all_preciseMedEfffalse_negatives) = [], [], [], []
    (all_approxMedEfftrue_positives_in_ground_truths, all_approxMedEfftrue_positives_in_model_annotations, all_approxMedEfffalse_positives,
     all_approxMedEfffalse_negatives) = [], [], [], []

    for chunk_id in chunk_list:
        human_annotations = get_all_human_annotations_for_chunk_id(chunk_id, check=True)
        pkl_file = os.path.join('outputs', 'model_pkls', f'{str(chunk_id)}_{model.model_name}_outputs.pickle')
        if rerun:
            query_a_model(model, get_chunk_filepath_from_chunk_id(chunk_id),
                          context_window, pkl_file)

        model_outputs = pickle.load(open(pkl_file, "rb", -1))
        check_errors(model_outputs, human_annotations, os.path.join('outputs', 'model_errors'), chunk_id, model.model_name)

        ### NER
        precise_NER_true_positives_in_ground_truths, precise_NER_true_positives_in_model_annotations, precise_NER_false_positives, precise_NER_false_negatives = NER_evaluation(
            model_outputs,
            human_annotations,
            abbreviated_precise_match)
        all_precise_NER_true_positives_in_ground_truths.extend(precise_NER_true_positives_in_ground_truths)
        all_precise_NER_true_positives_in_model_annotations.extend(precise_NER_true_positives_in_model_annotations)
        all_precise_NER_false_positives.extend(precise_NER_false_positives)
        all_precise_NER_false_negatives.extend(precise_NER_false_negatives)

        approx_NER_true_positives_in_ground_truths, approx_NER_true_positives_in_model_annotations, approx_NER_false_positives, approx_NER_false_negatives = NER_evaluation(
            model_outputs, human_annotations,
            abbreviated_approximate_match)
        all_approx_NER_true_positives_in_ground_truths.extend(approx_NER_true_positives_in_ground_truths)
        all_approx_NER_true_positives_in_model_annotations.extend(approx_NER_true_positives_in_model_annotations)
        all_approx_NER_false_positives.extend(approx_NER_false_positives)
        all_approx_NER_false_negatives.extend(approx_NER_false_negatives)

        ### Medical Conditions
        (preciseMedCondtrue_positives_in_ground_truths, preciseMedCondtrue_positives_in_model_annotations, preciseMedCondfalse_positives,
         preciseMedCondfalse_negatives) = RE_evaluation(model_outputs,
                                                        human_annotations,
                                                        'precise',
                                                        'medical_conditions')
        all_preciseMedCondtrue_positives_in_ground_truths.extend(preciseMedCondtrue_positives_in_ground_truths)
        all_preciseMedCondtrue_positives_in_model_annotations.extend(preciseMedCondtrue_positives_in_model_annotations)
        all_preciseMedCondfalse_positives.extend(preciseMedCondfalse_positives)
        all_preciseMedCondfalse_negatives.extend(preciseMedCondfalse_negatives)

        (approxMedCondtrue_positives_in_ground_truths, approxMedCondtrue_positives_in_model_annotations, approxMedCondfalse_positives,
         approxMedCondfalse_negatives) = RE_evaluation(model_outputs,
                                                       human_annotations,
                                                       'approximate',
                                                       'medical_conditions')
        all_approxMedCondtrue_positives_in_ground_truths.extend(approxMedCondtrue_positives_in_ground_truths)
        all_approxMedCondtrue_positives_in_model_annotations.extend(approxMedCondtrue_positives_in_model_annotations)
        all_approxMedCondfalse_positives.extend(approxMedCondfalse_positives)
        all_approxMedCondfalse_negatives.extend(approxMedCondfalse_negatives)

        ### Medicinal Effects
        (preciseMedEfftrue_positives_in_ground_truths, preciseMedEfftrue_positives_in_model_annotations, preciseMedEfffalse_positives,
         preciseMedEfffalse_negatives) = RE_evaluation(model_outputs,
                                                       human_annotations,
                                                       'precise',
                                                       'medicinal_effects')
        all_preciseMedEfftrue_positives_in_ground_truths.extend(preciseMedEfftrue_positives_in_ground_truths)
        all_preciseMedEfftrue_positives_in_model_annotations.extend(preciseMedEfftrue_positives_in_model_annotations)
        all_preciseMedEfffalse_positives.extend(preciseMedEfffalse_positives)
        all_preciseMedEfffalse_negatives.extend(preciseMedEfffalse_negatives)

        (approxMedEfftrue_positives_in_ground_truths, approxMedEfftrue_positives_in_model_annotations, approxMedEfffalse_positives,
         approxMedEfffalse_negatives) = RE_evaluation(model_outputs,
                                                      human_annotations,
                                                      'approximate',
                                                      'medicinal_effects')
        all_approxMedEfftrue_positives_in_ground_truths.extend(approxMedEfftrue_positives_in_ground_truths)
        all_approxMedEfftrue_positives_in_model_annotations.extend(approxMedEfftrue_positives_in_model_annotations)
        all_approxMedEfffalse_positives.extend(approxMedEfffalse_positives)
        all_approxMedEfffalse_negatives.extend(approxMedEfffalse_negatives)

    ### NER
    precise_NER_precision, precise_NER_recall, precise_NER_f1_score = get_metrics_from_tp_fp_fn(all_precise_NER_true_positives_in_ground_truths,
                                                                                                all_precise_NER_true_positives_in_model_annotations,
                                                                                                all_precise_NER_false_positives,
                                                                                                all_precise_NER_false_negatives)
    print(f'Precise NER')
    print(f'precision: {precise_NER_precision}, recall {precise_NER_recall}, f1:{precise_NER_f1_score}')

    approximate_NER_precision, approximate_NER_recall, approximate_NER_f1_score = get_metrics_from_tp_fp_fn(
        all_approx_NER_true_positives_in_ground_truths, all_approx_NER_true_positives_in_model_annotations, all_approx_NER_false_positives,
        all_approx_NER_false_negatives)

    print(f'Approximate NER')
    print(f'precision:{approximate_NER_precision}, recall:{approximate_NER_recall}, f1:{approximate_NER_f1_score}')

    ### Medical Conditions
    preciseMedCondprecision, preciseMedCondrecall, preciseMedCondf1_score = get_metrics_from_tp_fp_fn(
        all_preciseMedCondtrue_positives_in_ground_truths,
        all_preciseMedCondtrue_positives_in_model_annotations,
        all_preciseMedCondfalse_positives,
        all_preciseMedCondfalse_negatives)
    print(f'Precise MedCond')
    print(f'precision: {preciseMedCondprecision}, recall {preciseMedCondrecall}, f1:{preciseMedCondf1_score}')

    approximateMedCondprecision, approximateMedCondrecall, approximateMedCondf1_score = get_metrics_from_tp_fp_fn(
        all_approxMedCondtrue_positives_in_ground_truths, all_approxMedCondtrue_positives_in_model_annotations, all_approxMedCondfalse_positives,
        all_approxMedCondfalse_negatives)

    print(f'Approximate MedCond')
    print(f'precision:{approximateMedCondprecision}, recall:{approximateMedCondrecall}, f1:{approximateMedCondf1_score}')

    ### Medicinal Effects
    preciseMedEffprecision, preciseMedEffrecall, preciseMedEfff1_score = get_metrics_from_tp_fp_fn(
        all_preciseMedEfftrue_positives_in_ground_truths,
        all_preciseMedEfftrue_positives_in_model_annotations,
        all_preciseMedEfffalse_positives,
        all_preciseMedEfffalse_negatives)
    print(f'Precise MedEff')
    print(f'precision: {preciseMedEffprecision}, recall {preciseMedEffrecall}, f1:{preciseMedEfff1_score}')

    approximateMedEffprecision, approximateMedEffrecall, approximateMedEfff1_score = get_metrics_from_tp_fp_fn(
        all_approxMedEfftrue_positives_in_ground_truths, all_approxMedEfftrue_positives_in_model_annotations, all_approxMedEfffalse_positives,
        all_approxMedEfffalse_negatives)

    print(f'Approximate MedEff')
    print(f'precision:{approximateMedEffprecision}, recall:{approximateMedEffrecall}, f1:{approximateMedEfff1_score}')

    out_df = {'Precise NER': [precise_NER_precision, precise_NER_recall, precise_NER_f1_score],
              'Approximate NER': [approximate_NER_precision, approximate_NER_recall, approximate_NER_f1_score],
              'Precise MedCond': [preciseMedCondprecision, preciseMedCondrecall, preciseMedCondf1_score],
              'Approximate MedCond': [approximateMedCondprecision, approximateMedCondrecall, approximateMedCondf1_score],
              'Precise MedEff': [preciseMedEffprecision, preciseMedEffrecall, preciseMedEfff1_score],
              'Approximate MedEff': [approximateMedEffprecision, approximateMedEffrecall, approximateMedEfff1_score]}
    out_df = pd.DataFrame(out_df, index=['precision', 'recall', 'f1'])
    out_df.to_csv(os.path.join(out_dir, model.model_name + '_results.csv'))


def assessing_hparams():
    # This is a minimal process and more about getting the model to run and output something sensible than actual performance.
    from dotenv import load_dotenv

    load_dotenv(os.path.join(repo_path, 'MedicinalPlantMining', 'rag_models', '.env'))
    model1 = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
    assess_model_on_chunk_list(train['id'].unique().tolist(), model1, get_input_size_limit(16), 'hparam_runs', rerun=False)


def full_evaluation():
    all_models = setup_models()
    test = pd.read_csv(os.path.join('outputs', 'for_testing.csv'))
    for m in all_models:
        assess_model_on_chunk_list(test['id'].unique().tolist(), all_models[m][0], all_models[m][1], 'outputs')


def evaluate_on_all_papers():
    # May want to include full paper evaluations as an extra analysis.
    pass


def main():
    assessing_hparams()


if __name__ == '__main__':
    # _get_chunks_to_tweak_with()
    train = pd.read_csv(os.path.join('outputs', 'for_hparam_tuning.csv'))
    repo_path = os.environ.get('KEWSCRATCHPATH')
    base_text_path = os.path.join(repo_path, 'MedicinalPlantMining', 'annotated_data', 'top_10_medicinal_hits', 'text_files')
    base_chunk_path = os.path.join(repo_path, 'MedicinalPlantMining', 'annotated_data', 'top_10_medicinal_hits', 'chunks', 'all_chunks')
    main()
