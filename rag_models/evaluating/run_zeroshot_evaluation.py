import os
import sys

from langchain_openai import ChatOpenAI
sys.path.append("../../testing/evaluation_methods")
from rag_models.evaluating import NER_evaluation, precise_match, approximate_match
from rag_models.running_models import query_a_model, get_input_size_limit
from rag_models.structured_output_schema import get_all_human_annotations_for_corpus_id


def assessing_hparams():
    # Get API keys
    from dotenv import load_dotenv
    repo_path = os.environ.get('KEWSCRATCHPATH')
    load_dotenv(os.path.join(repo_path,'MedicinalPlantMining','rag_models', '.env'))

    base_text_path = os.path.join('..', '..', 'testing', 'test_medicinal_01', '10_medicinal_hits')
    corpus_id = '4187756'
    human_annotations = get_all_human_annotations_for_corpus_id(corpus_id)

    model1 = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
    gpt3_outputs = query_a_model(model1, os.path.join(base_text_path, f'{corpus_id}.txt'),
                                 get_input_size_limit(16))
    precise_gpt3_precision, precise_gpt3_recall, precise_gpt3_f1_score = NER_evaluation(gpt3_outputs, human_annotations, precise_match)
    approximate_gpt3_precision, approximate_gpt3_recall, approximate_gpt3_f1_score = NER_evaluation(gpt3_outputs, human_annotations,
                                                                                                    approximate_match)

    print(precise_gpt3_precision, precise_gpt3_recall, precise_gpt3_f1_score)

def full_evaluation():
    base_text_path = os.path.join('..', '..', 'testing', 'test_medicinal_01', '10_medicinal_hits')
    text_files = os.listdir(base_text_path)
    ids = [t.split('.txt')[0] for t in text_files]
    for corpus_id in ids:
        human_annotations = get_all_human_annotations_for_corpus_id(corpus_id)

        model1 = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
        gpt3_outputs = query_a_model(model1, os.path.join(base_text_path, f'{corpus_id}.txt'),
                                     get_input_size_limit(16))
        precise_gpt3_precision, precise_gpt3_recall, precise_gpt3_f1_score = NER_evaluation(gpt3_outputs, human_annotations, precise_match)
        approximate_gpt3_precision, approximate_gpt3_recall, approximate_gpt3_f1_score = NER_evaluation(gpt3_outputs, human_annotations,
                                                                                                        approximate_match)


def main():
    assessing_hparams()


if __name__ == '__main__':
    main()
