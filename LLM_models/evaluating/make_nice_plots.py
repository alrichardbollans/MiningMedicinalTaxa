import os

import pandas as pd
import seaborn as sns

from LLM_models.evaluating import compare_errors
from LLM_models.evaluating.run_evaluation import basic_plot_results


def collect_main_results():
    all_results = pd.DataFrame()
    fileNames = os.listdir(os.path.join('outputs', 'full_eval'))
    for f in fileNames:
        if f.endswith(".csv") and not f.startswith('all_results'):
            model_results = pd.read_csv(os.path.join('outputs', 'full_eval', f), index_col=0)
            model_results = model_results.loc[['f1']]
            model_results = model_results.rename(index={'f1': f'{f}_f1'})
            all_results = pd.concat([all_results, model_results])
    all_results.loc['model_means'] = all_results.fillna(0).mean(numeric_only=True)
    all_results = all_results.sort_values(by='Precise NER', ascending=False)
    all_results.to_csv(os.path.join(os.path.join('outputs', 'full_eval', 'all_results.csv')))


def melt_all_results(metric, models, _measures):
    renaming = {'claude-3-5-sonnet-20241022_autoremove_non_sci_names_results.csv': 'Claude_NS', 'claude-3-5-sonnet-20241022_results.csv': 'Claude',
                'deepseek-chat_autoremove_non_sci_names_results.csv': 'DeepSeek_NS', 'deepseek-chat_results.csv': 'DeepSeek',
                'gnfinder_autoremove_non_sci_names_results.csv': 'GNfinder_NS', 'gnfinder_results.csv': 'GNfinder',
                'gpt-4o-2024-08-06_autoremove_non_sci_names_results.csv': 'GPT_NS',
                'gpt-4o-2024-08-06_results.csv': 'GPT',
                'llama-v3p1-405b-instruct_autoremove_non_sci_names_results.csv': 'Llama_NS',
                'llama-v3p1-405b-instruct_results.csv': 'Llama',
                'en_ner_eco_biobert_autoremove_non_sci_names_results.csv': 'TaxoNERD_NS',
                'en_ner_eco_biobert_results.csv': 'TaxoNERD',
                'ft_gpt-4o-2024-08-06_personal__BHfNoQa3_results.csv': 'FTGPT',
                'ft_gpt-4o-2024-08-06_personal__BHfNoQa3_autoremove_non_sci_names_results.csv': 'FTGPT_NS'}
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


def for_full_eval(models, _measures, file_tag: str, inc_legend: bool = True):
    for metric in metrics:
        all_results = melt_all_results(metric, models, _measures)
        all_results = all_results[all_results['Model'].isin(models)]
        all_results = all_results[all_results['class'].isin(_measures)]

        all_results['class'] = all_results['class'].apply(lambda x: x.replace('NER', 'SNER').replace('Precise', 'Exact').replace('Approx.', 'Relaxed'))

        import matplotlib.pyplot as plt

        if file_tag=='RE':
            col_wrap = 2
        else:
            col_wrap=None

        sns.set_theme(style="whitegrid")
        g = sns.catplot(
            data=all_results, x="Model", y=metric, col="class", hue='NS', col_wrap=col_wrap,
            kind="bar", height=4, aspect=.6, palette=["#E98F66","#53B68B"]
        )
        g.set_axis_labels("", metric)

        for i in g.axes.flat:
            i.yaxis.label.set(rotation='horizontal', ha='right')
        g.set_titles("{col_name}")
        if col_wrap is None:
            g.set_xticklabels(rotation=45, ha='right', rotation_mode='anchor')
        else:
            g.set_xticklabels(g.axes.flat[-1].get_xticklabels(),rotation=45, ha='right', rotation_mode='anchor')
        g.set(ylim=(0, 1))
        g.despine(left=True)
        plt.tight_layout()

        if inc_legend:
            print('inc legend')
            sns.move_legend(g,loc="center left", ncol=1, bbox_to_anchor=(1, 0.5), title='Cleaned\n Names')
        else:
            g.legend.remove()

        plt.savefig(os.path.join('outputs', 'full_eval', 'compiled_results', f'{file_tag}_{metric}_results.png'), dpi=300, bbox_inches="tight")
        plt.close()
        all_results.to_csv(os.path.join(os.path.join('outputs', 'full_eval', 'compiled_results', f'{file_tag}_{metric}_results.csv')))
def plots():
    basic_plot_results(os.path.join('outputs', 'full_eval', 'gpt-4o-2024-08-06_results.csv'), os.path.join('outputs', 'full_eval'), 'gpt-4o-2024-08-06')
    basic_plot_results(os.path.join('outputs', 'full_eval', 'ft_gpt-4o-2024-08-06_personal__BHfNoQa3_results.csv'), os.path.join('outputs', 'full_eval'), 'ft_gpt-4o-2024-08-06_personal__BHfNoQa3')

    # collect_main_results()
    #
    # #
    # ## NER
    # _models = ['Claude', 'DeepSeek', 'GNfinder', 'GPT', 'Llama', 'TaxoNERD']
    # measures = ['Precise NER', 'Approx. NER']
    # for_full_eval(_models, measures, 'NER')
    #
    # # ## RE
    # _models = ['Claude', 'DeepSeek', 'GPT', 'Llama']
    # measures = ['Precise MedCond', 'Approx. MedCond', 'Precise MedEff', 'Approx. MedEff']
    # for_full_eval(_models, measures, 'RE')
    # for_full_eval(_models, ['Precise MedCond', 'Approx. MedCond'], 'MedCond')
    # for_full_eval(_models, ['Precise MedEff', 'Approx. MedEff'], 'MedEff', inc_legend=False)
    #
    # ## Finetuning
    # _models = ['GPT', 'FTGPT']
    # for_full_eval(_models, all_measures, 'FT')

if __name__ == '__main__':
    all_measures = ['Precise NER', 'Approx. NER', 'Precise MedCond', 'Approx. MedCond', 'Precise MedEff', 'Approx. MedEff']
    all_models = ['Claude', 'DeepSeek', 'GNfinder', 'GPT', 'FTGPT', 'Llama', 'TaxoNERD']
    metrics = ['f1', 'precision', 'recall']

    plots()

