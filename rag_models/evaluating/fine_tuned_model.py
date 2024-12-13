import os

import pandas as pd

from rag_models.evaluating.run_evaluation import assess_model_on_chunk_list
from rag_models.running_models import get_input_size_limit
from rag_models.structured_output_schema import repo_path


def get_fine_tuned_model():
    from langchain_openai import ChatOpenAI

    # Get API keys
    from dotenv import load_dotenv

    load_dotenv(os.path.join(repo_path, 'MedicinalPlantMining', 'rag_models', '.env'))
    out = {}

    hparams = {'temperature': 0}

    # Max tokens 128k
    # Input: $3.750 /1M tokens
    # Output $15.00 /1M tokens
    model1 = ChatOpenAI(model="ft:gpt-4o-2024-08-06:personal::Acwijdma", **hparams)
    out['gpt4o_FT'] = [model1, get_input_size_limit(128)]
    return out


def assessing_hparams(rerun: bool = True):

    all_models = get_fine_tuned_model()

    for m in all_models:
        print(m)
        assess_model_on_chunk_list(df_for_hparam_tuning['id'].unique().tolist(), all_models[m][0], all_models[m][1], 'hparam_runs', rerun=rerun,
                                   autoremove_non_sci_names=False)
        model_results = assess_model_on_chunk_list(df_for_hparam_tuning['id'].unique().tolist(), all_models[m][0], all_models[m][1],
                                                   'hparam_runs', rerun=False, autoremove_non_sci_names=True)
        model_results = model_results.loc[['f1']]
        model_results = model_results.rename(index={'f1': f'{m}_f1'})


def full_evaluation(rerun: bool = True):

    all_models = get_fine_tuned_model()
    test = pd.read_csv(os.path.join('outputs', 'for_testing.csv'))
    for m in all_models:
        print(m)
        assess_model_on_chunk_list(test['id'].unique().tolist(), all_models[m][0], all_models[m][1], os.path.join('outputs', 'full_eval'),
                                   rerun=rerun,
                                   autoremove_non_sci_names=False)
        assess_model_on_chunk_list(test['id'].unique().tolist(), all_models[m][0], all_models[m][1], os.path.join('outputs', 'full_eval'),
                                   rerun=False,
                                   autoremove_non_sci_names=True)

if __name__ == '__main__':
    df_for_hparam_tuning = pd.read_csv(os.path.join('outputs', 'for_hparam_tuning.csv'))
    # assessing_hparams()
    full_evaluation()
    # main()
