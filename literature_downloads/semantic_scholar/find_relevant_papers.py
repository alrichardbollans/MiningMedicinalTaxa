import os
import time

import numpy as np
import pandas as pd
import requests

download_path = os.path.join('downloads')
abstracts_path = os.path.join(download_path, 'abstracts')
paper_info_path = os.path.join(download_path, 'paper_info')

# Request API key from Semantic Scholar and save in .env file
import dotenv

dotenv.load_dotenv()
S2_API_KEY = os.environ['S2_API_KEY']
headers = {
    'x-api-key': S2_API_KEY,
}


def download_paper_info_and_abstracts(query):
    paper_df = pd.DataFrame()
    result_limit = 100
    params = {'query': query, 'limit': result_limit, 'user-agent': 'requests/2.0.0',
              'fields': 'title,year,abstract,authors,isOpenAccess,url,externalIds', 'offset': 0}
    while True:
        rsp = requests.get('https://api.semanticscholar.org/graph/v1/paper/search',
                           params=params, headers=headers)
        rsp.raise_for_status()
        results = rsp.json()
        total = results["total"]
        if not total:
            print('No matches found. Please try another query.')

        print(f'Found {total} results. Showing up to {result_limit} with offset {params["offset"]}.')
        papers = results['data']
        ids = []
        abstracts = {}

        for p in papers:
            ids.append(p['paperId'])
            abstracts[p['paperId']] = p['abstract']
            authors = ', '.join([x['name'] for x in p['authors']])
            if 'DOI' in p['externalIds']:
                doi = p['externalIds']['DOI']
            else:
                doi = np.nan
            info_df = pd.DataFrame(
                {'paperId': [p['paperId']], 'DOI': [doi],
                 'title': [p['title']], 'year': [p['year']], 'authors': [authors],
                 'isOpenAccess': [p['isOpenAccess']], 'url': [p['url']],
                 'abstract_path': [os.path.join(abstracts_path, p['paperId'] + '.txt')]})
            paper_df = pd.concat([paper_df, info_df])

        for p_id in abstracts:
            abstract = abstracts[p_id]
            if abstract is not None:
                f = open(os.path.join(abstracts_path, p_id + '.txt'), 'w')
                f.write(abstract)
                f.close()

        if len(papers) < result_limit:
            break
        params['offset'] += len(papers)
    paper_df.to_csv(os.path.join(paper_info_path, query + '.csv'))

    time.sleep(float(1 / 90))


if __name__ == '__main__':
    download_paper_info_and_abstracts('aspidosperma antiplasmodial')

    # for f in search_filtering_terms:
    #     download_paper_info_and_abstracts(f)
