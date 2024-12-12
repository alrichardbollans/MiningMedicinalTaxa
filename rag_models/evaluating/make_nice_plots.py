import os

import pandas as pd
import seaborn as sns


def for_full_eval():
    renaming = {'claude-3-5-sonnet-20241022_autoremove_non_sci_names_results.csv': 'Claude_NS', 'claude-3-5-sonnet-20241022_results.csv': 'Claude',
                'gemini-1.5-pro-002_autoremove_non_sci_names_results.csv': 'Gemini_NS', 'gemini-1.5-pro-002_results.csv': 'Gemini',
                'gnfinder_autoremove_non_sci_names_results.csv': 'Gnfinder_NS', 'gnfinder_results.csv': 'Gnfinder', 'gpt-4o_autoremove_non_sci_names_results.csv': 'GPT_NS',
                'gpt-4o_results.csv': 'GPT', 'llama-v3p1-405b-instruct_autoremove_non_sci_names_results.csv': 'Llama_NS',
                'llama-v3p1-405b-instruct_results.csv': 'Llama'}
    fileNames = os.listdir(os.path.join('outputs', 'full_eval'))
    for metric in metrics:
        all_results = pd.DataFrame()
        for f in fileNames:
            if f.endswith(".csv") and not f.startswith('all_results'):

                model_results = pd.read_csv(os.path.join('outputs', 'full_eval', f), index_col=0)

                metric_results = model_results.loc[[metric]].T
                metric_results['class'] = metric_results.index
                metric_results['Model'] = renaming[f]

                all_results = pd.concat([all_results, metric_results])
        # print(str(renaming))
        # all_results = all_results.sort_values(by='Precise NER', ascending=False)
        # all_results['Metric'] = metric
        all_results = all_results.reset_index(drop=True)
        all_results.to_csv(os.path.join(os.path.join('outputs', 'full_eval', 'compiled_results', f'{metric}_results.csv')))
        # all_results = all_results.set_index(keys=['Model'])

        # melted_df = all_results.melt()
        # df = sns.load_dataset("titanic")
        import matplotlib.pyplot as plt
        g = sns.catplot(
            data=all_results, x="Model", y=metric, col="class",
            kind="bar", height=4, aspect=.6,
        )
        g.set_axis_labels("", metric)
        g.set_titles("{col_name}")
        g.set_xticklabels(rotation=45, ha='right')
        # g.set(ylim=(0, 1))
        g.despine(left=True)
        plt.tight_layout()
        plt.savefig(os.path.join('outputs', 'full_eval', 'compiled_results', f'{metric}_results.png'), dpi=300)
        plt.close()


if __name__ == '__main__':
    metrics = ['f1', 'precision', 'recall']
    measures = ['Precise NER', 'Approx. NER', 'Precise MedCond', 'Approx. MedCond', 'Precise MedEff', 'Approx. MedEff']

    for_full_eval()
