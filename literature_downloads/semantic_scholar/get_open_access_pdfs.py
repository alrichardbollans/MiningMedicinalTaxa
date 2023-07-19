import os
import time
from typing import Generator, Union, List, Tuple

import pandas as pd
import urllib3
from requests import Session

from literature_downloads.semantic_scholar import download_path, paper_info_path, S2_API_KEY

# urllib3.disable_warnings()


downloaded_papers = os.path.join(download_path, 'papers_pdfs')


def get_paper(session: Session, paper_id: str, fields: str = 'paperId,title', **kwargs) -> dict:
    params = {
        'fields': fields,
        **kwargs,
    }
    headers = {
        'x-api-key': S2_API_KEY,
    }

    with session.get(f'https://api.semanticscholar.org/graph/v1/paper/{paper_id}', params=params,
                     headers=headers) as response:
        response.raise_for_status()
        time.sleep(float(1 / 90))

        return response.json()


def download_pdf(session: Session, url: str, path: str, user_agent: str = 'requests/2.0.0'):
    # send a user-agent to avoid server error
    headers = {
        'user-agent': user_agent,
    }

    # stream the response to avoid downloading the entire file into memory
    with session.get(url, headers=headers, stream=True, verify=False) as response:
        # check if the request was successful
        response.raise_for_status()

        if response.headers['content-type'] != 'application/pdf':
            raise Exception('The response is not a pdf')

        with open(path, 'wb') as f:
            # write the response to the file, chunk_size bytes at a time
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        time.sleep(float(1 / 90))


def download_paper(session: Session, paper_id: str, directory: str = 'papers_pdfs',
                   user_agent: str = 'requests/2.0.0') -> Union[str, None]:
    paper = get_paper(session, paper_id, fields='paperId,isOpenAccess,openAccessPdf')

    # check if the paper is open access
    if not paper['isOpenAccess']:
        return None

    paperId: str = paper['paperId']
    pdf_url: str = paper['openAccessPdf']['url']
    pdf_path = os.path.join(directory, f'{paperId}.pdf')

    # create the directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)

    # check if the pdf has already been downloaded
    if not os.path.exists(pdf_path):
        download_pdf(session, pdf_url, pdf_path, user_agent=user_agent)

    return pdf_path


def download_papers(paper_ids: List[str], directory: str = 'papers_pdfs',
                    user_agent: str = 'requests/2.0.0') -> \
        Generator[Tuple[str, Union[str, None, Exception]], None, None]:
    # use a session to reuse the same TCP connection
    with Session() as session:
        for paper_id in paper_ids:
            try:
                yield paper_id, download_paper(session, paper_id, directory=directory, user_agent=user_agent)
            except Exception as e:
                yield paper_id, e


def main(paper_ids, directory, user_agent) -> None:
    for paper_id, result in download_papers(paper_ids, directory=directory,
                                            user_agent=user_agent):
        if isinstance(result, Exception):
            print(f"Failed to download '{paper_id}': {type(result).__name__}: {result}")
        elif result is None:
            print(f"'{paper_id}' is not open access")
        else:
            print(f"Downloaded '{paper_id}' to '{result}'")


if __name__ == '__main__':
    initial_query = 'aspidosperma antiplasmodial'
    info_df = pd.read_csv(os.path.join(paper_info_path, initial_query + '.csv'))
    info_df = info_df[info_df['isOpenAccess']]
    paper_ids = info_df['paperId'].unique().tolist()
    main(paper_ids, os.path.join(downloaded_papers, initial_query), user_agent='requests/2.0.0')
