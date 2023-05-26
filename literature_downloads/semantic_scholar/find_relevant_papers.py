import os

import pandas as pd
import requests

download_path = os.path.join('downloads')
abstracts_path = os.path.join(download_path, 'abstracts')
paper_info_path = os.path.join(download_path, 'paper_info')


# import dotenv
# dotenv.load_dotenv()
# S2_API_KEY = os.environ['S2_API_KEY']
# headers = {
#         'x-api-key': S2_API_KEY,
#     }

def download_paper_info_and_abstracts(query):
    result_limit = 100
    # modify offset to get all results up to 'total'
    rsp = requests.get('https://api.semanticscholar.org/graph/v1/paper/search',
                       params={'query': query, 'limit': result_limit, 'user-agent': 'requests/2.0.0',
                               'fields': 'title,year,abstract,authors,isOpenAccess'})
    rsp.raise_for_status()
    results = rsp.json()
    total = results["total"]
    if not total:
        print('No matches found. Please try another query.')

    print(f'Found {total} results. Showing up to {result_limit}.')
    papers = results['data']
    ids = []
    abstracts = {}

    paper_df = pd.DataFrame()

    for p in papers:
        ids.append(p['paperId'])
        abstracts[p['paperId']] = p['abstract']
        authors = ', '.join([x['name'] for x in p['authors']])
        info_df = pd.DataFrame(
            {'paperId': [p['paperId']], 'title': [p['title']], 'year': [p['year']], 'authors': [authors],
             'isOpenAccess': [p['isOpenAccess']]})
        paper_df = pd.concat([paper_df, info_df])

    paper_df.to_csv(os.path.join(paper_info_path, query + '.csv'))

    for p_id in abstracts:
        abstract = abstracts[p_id]
        if abstract is not None:
            f = open(os.path.join(abstracts_path, p_id + '.txt'), 'w')
            f.write(abstract)
            f.close()


if __name__ == '__main__':
    download_paper_info_and_abstracts('grass antiplasmodial')
