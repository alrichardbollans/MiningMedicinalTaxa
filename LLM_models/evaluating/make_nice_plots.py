import os

import pandas as pd
import seaborn as sns


def melt_all_results(metric,models,_measures):
    renaming = {'claude-3-5-sonnet-20241022_autoremove_non_sci_names_results.csv': 'Claude_NS', 'claude-3-5-sonnet-20241022_results.csv': 'Claude',
                'gemini-1.5-pro-002_autoremove_non_sci_names_results.csv': 'Gemini_NS', 'gemini-1.5-pro-002_results.csv': 'Gemini',
                'gnfinder_autoremove_non_sci_names_results.csv': 'Gnfinder_NS', 'gnfinder_results.csv': 'Gnfinder',
                'gpt-4o_autoremove_non_sci_names_results.csv': 'GPT_NS',
                'gpt-4o_results.csv': 'GPT', 'llama-v3p1-405b-instruct_autoremove_non_sci_names_results.csv': 'Llama_NS',
                'llama-v3p1-405b-instruct_results.csv': 'Llama',
                'ftgpt-4o-2024-08-06personalAcwijdma_results.csv': 'FTGPT',
                'ftgpt-4o-2024-08-06personalAcwijdma_autoremove_non_sci_names_results.csv': 'FTGPT_NS'}
    fileNames = os.listdir(os.path.join('outputs', 'full_eval'))
    all_results = pd.DataFrame()
    for f in fileNames:
        if f.endswith(".csv") and not f.startswith('all_results'):
            model_results = pd.read_csv(os.path.join('outputs', 'full_eval', f), index_col=0)

            metric_results = model_results.loc[[metric]].T
            metric_results['class'] = metric_results.index

            metric_results['Model'] = renaming[f]

            all_results = pd.concat([all_results, metric_results])

    all_results = all_results.reset_index(drop=True)
    all_results['NS'] = all_results['Model'].str.contains('_NS')
    all_results['Model'] = all_results['Model'].str.replace('_NS', '')
    all_results = all_results[all_results['Model'].isin(models)]
    all_results = all_results[all_results['class'].isin(_measures)]

    model_sort_order = [c for c in all_models if c in models]
    measure_sort_order = [c for c in all_measures if c in _measures]
    all_results['Model'] = pd.Categorical(all_results['Model'], ordered=True,
                              categories=model_sort_order)
    all_results['class'] = pd.Categorical(all_results['class'], ordered=True,
                                          categories=measure_sort_order)

    all_results = all_results.sort_values(by=['Model', 'NS', 'class'])
    return all_results

def for_full_eval(models,_measures, file_tag:str):

    for metric in metrics:
        all_results = melt_all_results(metric,models, _measures)
        all_results = all_results[all_results['Model'].isin(models)]
        all_results = all_results[all_results['class'].isin(_measures)]
        import matplotlib.pyplot as plt
        sns.set_theme(style="whitegrid", palette="colorblind")
        g = sns.catplot(
            data=all_results, x="Model", y=metric, col="class",hue='NS',
            kind="bar", height=4, aspect=.6#, palette=["b", "m"]
        )
        g.set_axis_labels("", metric)

        g.axes.flat[0].yaxis.label.set(rotation='horizontal', ha='right')
        g.set_titles("{col_name}")
        g.set_xticklabels(rotation=45, ha='right', rotation_mode='anchor')
        g.set(ylim=(0, 1))
        g.despine(left=True)
        g.legend.remove()
        # g.legend.set_title("AutoRemove")
        plt.tight_layout()
        plt.savefig(os.path.join('outputs', 'full_eval', 'compiled_results', f'{file_tag}_{metric}_results.png'), dpi=300)
        plt.close()
        all_results.to_csv(os.path.join(os.path.join('outputs', 'full_eval', 'compiled_results', f'{file_tag}_{metric}_results.csv')))



if __name__ == '__main__':
    metrics = ['f1', 'precision', 'recall']
    all_measures = ['Precise NER', 'Approx. NER', 'Precise MedCond', 'Approx. MedCond', 'Precise MedEff', 'Approx. MedEff']
    all_models = ['Claude', 'Gemini', 'Gnfinder', 'GPT', 'FTGPT', 'Llama']

    ## NER
    _models = ['Claude', 'Gemini', 'Gnfinder', 'GPT', 'Llama']
    measures = ['Precise NER', 'Approx. NER']
    for_full_eval(_models,measures, 'NER')

    ## RE
    _models = ['Claude', 'Gemini', 'GPT', 'Llama']
    measures = ['Precise MedCond', 'Approx. MedCond', 'Precise MedEff', 'Approx. MedEff']
    for_full_eval(_models, measures, 'RE')

    ## Finetuning
    _models = ['GPT', 'FTGPT']
    for_full_eval(_models, all_measures, 'FT')