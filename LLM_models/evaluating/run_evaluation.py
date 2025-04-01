import os
import pickle
import time
from collections.abc import Callable

import pandas as pd
from pydantic.v1 import BaseModel

from LLM_models.evaluating import NER_evaluation, RE_evaluation, check_errors, abbreviated_approximate_match, \
    abbreviated_precise_match, get_metrics_from_tp_fp_fn, clean_model_annotations_using_taxonomy_knowledge
from LLM_models.evaluating.gnfinder_baseline import gnfinder_query_function
from LLM_models.running_models import query_a_model, get_input_size_limit, setup_models
from LLM_models.structured_output_schema import valid_chunk_annotation_info, get_all_human_annotations_for_chunk_id, get_chunk_filepath_from_chunk_id, \
    repo_path, summarise_annotations


def _get_chunks_to_tweak_with():
    raise ValueError('Are you sure you want to redefine?')
    ''' Splits into train/test chunks but will be easiest to use whole papers instead.'''
    from sklearn.model_selection import train_test_split

    assert valid_chunk_annotation_info['reference_only'].unique().tolist() == ['no']

    for_testing, for_hparam_tuning = train_test_split(valid_chunk_annotation_info, test_size=0.1, shuffle=True)

    assert len(for_testing) > len(for_hparam_tuning)

    for i in for_hparam_tuning['id'].values:
        assert i not in for_testing['id'].values

    for_hparam_tuning.to_csv(os.path.join('outputs', 'for_hparam_tuning.csv'), index=False)
    for_testing.to_csv(os.path.join('outputs', 'for_testing.csv'), index=False)


def ___get_train_test_papers():
    raise ValueError('Are you sure you want to redefine?')

    # Not going to use this now
    ids = valid_chunk_annotation_info['corpus_id'].unique().tolist()
    # assert len(ids) == 10
    train = ['4187756']  # just use first paper
    test = [c for c in ids if c not in train]
    return train, test


def clean_model_name(model_name: str) -> str:
    if '/' in model_name:
        model_name = model_name[model_name.rindex('/') + 1:]
    model_name = model_name.replace(':', '_')
    return model_name


def assess_model_on_chunk_list(chunk_list, model, context_window, out_dir, rerun: bool = True, autoremove_non_sci_names: bool = False,
                               model_query_function: Callable = None):
    """
    Given a list of chunk ids, runs the model on each chunk and collects outputs


    :param chunk_list:
    :param model:
    :param context_window:
    :param out_dir:
    :param rerun: Whether to query the model again, or used cached results
    :param autoremove_non_sci_names: Whether to use taxonomy knowledge to remove non-scientific name hits from the model outputs
    :param model_query_function: Function to call to query the model
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
    try:
        model_name = model.model_name
    except AttributeError:
        # Different models use different fields for model names for some reason
        model_name = model.model

    model_name = clean_model_name(model_name)

    def run(c_id, p_file):
        print(c_id)
        time.sleep(0.5)
        if model_query_function is None:
            query_a_model(model, get_chunk_filepath_from_chunk_id(c_id),
                          context_window, p_file)
        else:
            model_query_function(model, get_chunk_filepath_from_chunk_id(c_id),
                                 context_window, p_file)
        m_outputs = pickle.load(open(p_file, "rb", -1))
        return m_outputs

    model_tag = model_name
    if autoremove_non_sci_names:
        model_tag += '_autoremove_non_sci_names'
    for chunk_id in chunk_list:
        pkl_file = os.path.join('outputs', 'model_pkls', f'{str(chunk_id)}_{model_name}_outputs.pickle')
        if rerun:
            model_outputs = run(chunk_id, pkl_file)
        else:
            try:
                model_outputs = pickle.load(open(pkl_file, "rb", -1))
            except FileNotFoundError:
                print(f'With rerun=False, cant find associated pkl file: {pkl_file}. Rerunning')
                model_outputs = run(chunk_id, pkl_file)

        if autoremove_non_sci_names:
            model_outputs = clean_model_annotations_using_taxonomy_knowledge(model_outputs)

        human_annotations = get_all_human_annotations_for_chunk_id(chunk_id, check=True)
        check_errors(model_outputs, human_annotations, os.path.join('outputs', 'model_errors'), chunk_id, model_tag)

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
              'Approx. NER': [approximate_NER_precision, approximate_NER_recall, approximate_NER_f1_score],
              'Precise MedCond': [preciseMedCondprecision, preciseMedCondrecall, preciseMedCondf1_score],
              'Approx. MedCond': [approximateMedCondprecision, approximateMedCondrecall, approximateMedCondf1_score],
              'Precise MedEff': [preciseMedEffprecision, preciseMedEffrecall, preciseMedEfff1_score],
              'Approx. MedEff': [approximateMedEffprecision, approximateMedEffrecall, approximateMedEfff1_score]}

    out_df = pd.DataFrame(out_df, index=['precision', 'recall', 'f1'])
    out_df.to_csv(os.path.join(out_dir, model_tag + '_results.csv'))
    basic_plot_results(os.path.join(out_dir, model_tag + '_results.csv'), out_dir, model_tag)
    return out_df, all_approxMedCondtrue_positives_in_model_annotations


def basic_plot_results(file_to_plot, out_dir, model_name):
    import matplotlib.pyplot as plt
    import seaborn as sns

    out_df = pd.read_csv(file_to_plot, index_col=0)
    rename_dict = {}
    for c in out_df.columns:
        rename_dict[c] = c.replace('NER', 'SNER').replace('Precise', 'Exact').replace('Approx.', 'Relaxed')
    out_df.rename(columns=rename_dict, inplace=True)

    sns.heatmap(out_df, annot=True, cmap='inferno')
    plt.xticks(rotation=45, ha='right', rotation_mode='anchor')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, model_name + '_results.png'), dpi=300)
    plt.close()


def plot_results(file_info, out_dir):
    import matplotlib.pyplot as plt
    import seaborn as sns

    pairs = ['Precise NER', 'Approx. NER', 'Precise MedCond', 'Approx. MedCond', 'Precise MedEff', 'Approx. MedEff']
    for p in pairs:
        df = pd.DataFrame()
        for model_name in file_info:
            out_df = pd.read_csv(file_info[model_name][0], index_col=0)[[p]]
            out_df = out_df.rename(columns={p: model_name})
            df = pd.concat([df, out_df], axis=1)
        sns.heatmap(df, annot=True, cmap='inferno')
        plt.title(p)

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, p + '_results.png'), dpi=300)
        plt.close()


def assessing_hparams(rerun: bool = True):
    from langchain_openai import ChatOpenAI

    # This is a minimal process and more about getting the model to run and output something sensible than actual performance.
    from dotenv import load_dotenv

    load_dotenv(os.path.join(repo_path, 'MedicinalPlantMining', 'LLM_models', '.env'))

    all_models = setup_models()

    all_results = pd.DataFrame()
    for m in all_models:
        print(m)
        assess_model_on_chunk_list(df_for_hparam_tuning['id'].unique().tolist(), all_models[m][0], all_models[m][1], 'hparam_runs', rerun=rerun,
                                   autoremove_non_sci_names=False)
        model_results = assess_model_on_chunk_list(df_for_hparam_tuning['id'].unique().tolist(), all_models[m][0], all_models[m][1],
                                                   'hparam_runs', rerun=False, autoremove_non_sci_names=True)[0]
        model_results = model_results.loc[['f1']]
        model_results = model_results.rename(index={'f1': f'{m}_f1'})
        all_results = pd.concat([all_results, model_results])

    all_results.loc['model_means'] = all_results.fillna(0).mean(numeric_only=True)
    all_results.to_csv(os.path.join(os.path.join('hparam_runs', 'all_results.csv')))


def full_evaluation(rerun: bool = True):
    from dotenv import load_dotenv

    load_dotenv(os.path.join(repo_path, 'MedicinalPlantMining', 'LLM_models', '.env'))

    test = pd.read_csv(os.path.join('outputs', 'for_testing.csv'))

    all_models = setup_models()
    # all_models = {'gpt4o': [Mplaceholder(model_name='gpt-4o'), None],
    #               'gemini': [Mplaceholder(model_name='gemini-1.5-pro-002'), None],
    #               'claude': [Mplaceholder(model_name='claude-3-5-sonnet-20241022'), None],
    #               'llama': [Mplaceholder(model_name='llama-v3p1-405b-instruct'), None]}

    for m in all_models:
        print(m)
        assess_model_on_chunk_list(test['id'].unique().tolist(), all_models[m][0], all_models[m][1], os.path.join('outputs', 'full_eval'),
                                   rerun=rerun,
                                   autoremove_non_sci_names=False)
        assess_model_on_chunk_list(test['id'].unique().tolist(), all_models[m][0], all_models[m][1], os.path.join('outputs', 'full_eval'),
                                   rerun=False,
                                   autoremove_non_sci_names=True)


def full_eval_gnfinder(rerun: bool = True):
    test = pd.read_csv(os.path.join('outputs', 'for_testing.csv'))
    model = Mplaceholder(model_name='gnfinder')
    assess_model_on_chunk_list(test['id'].unique().tolist(), model, None, os.path.join('outputs', 'full_eval'), rerun=rerun,
                               autoremove_non_sci_names=False, model_query_function=gnfinder_query_function)
    assess_model_on_chunk_list(test['id'].unique().tolist(), model, None, os.path.join('outputs', 'full_eval'), rerun=False,
                               autoremove_non_sci_names=True, model_query_function=gnfinder_query_function)


def main():
    # assessing_hparams(rerun=True)
    full_evaluation(rerun=True)
    # full_eval_gnfinder(rerun=True)


if __name__ == '__main__':
    # _get_chunks_to_tweak_with()
    class Mplaceholder(BaseModel):
        """Extracted data about taxa."""

        # Creates a model so that we can extract multiple entities.
        model_name: str


    df_for_hparam_tuning = pd.read_csv(os.path.join('outputs', 'for_hparam_tuning.csv'))
    main()
