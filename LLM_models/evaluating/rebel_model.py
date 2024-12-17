import pickle
import spacy
import spacy_component  # Downloaded from https://github.com/Babelscape/rebel/blob/main/spacy_component.py
from LLM_models.loading_files import get_txt_from_file

from LLM_models.structured_output_schema import Taxon, deduplicate_and_standardise_output_taxa_lists


def get_rebel_output_on_text_file(text_file: str):
    nlp = spacy.load("en_core_web_sm")

    nlp.add_pipe("rebel", after="senter", config={
        'device': 0,  # Number of the GPU, -1 if want to use CPU
        'model_name': 'Babelscape/rebel-large'}  # Model used, will default to 'Babelscape/rebel-large' if not given
                 )
    input_sentence = get_txt_from_file(text_file)
    doc = nlp(input_sentence)
    doc_list = nlp.pipe([input_sentence])
    return doc


def convert_rebel_output_to_TaxaData(rebel_output):
    # Relations are from wikidata linking
    # Relations are here https://github.com/Babelscape/rebel/blob/54ea5fd07dafece420c28c6f71f1c6431f42797c/data/relations_count.tsv#L227
    # Relevant relations are:
    # - 'medical condition treated'
    # - 'drug used for treatment'
    # - maybe 'has effect'
    relevant_relations = ['medical condition treated', 'drug used for treatment', 'has effect']
    # (0, 6): {'relation': 'medical condition treated', 'head_span': Cinchona calisaya, 'tail_span': malaria}
    # (6, 0): {'relation': 'drug used for treatment', 'head_span': malaria, 'tail_span': Cinchona calisaya}
    # (18, 11): {'relation': 'drug used for treatment', 'head_span': fever, 'tail_span': Aspidosperma pubsecens L.}
    taxa_list = []
    for value, rel_dict in rebel_output._.rel.items():
        if rel_dict['relation'] in relevant_relations:
            if rel_dict['relation'] == 'has effect':
                taxon = rel_dict['tail_span']
                effect = rel_dict['head_span']
                taxa_list.append(Taxon(scientific_name=taxon, medicinal_effects=[effect]))
            else:
                if rel_dict['relation'] == 'medical condition treated':
                    taxon = rel_dict['head_span']
                    condition = rel_dict['tail_span']
                elif rel_dict['relation'] == 'drug used for treatment':
                    taxon = rel_dict['tail_span']
                    condition = rel_dict['head_span']
                taxa_list.append(Taxon(scientific_name=taxon, medical_conditions=[condition]))

    out_taxa = deduplicate_and_standardise_output_taxa_lists(taxa_list)

    return out_taxa


def rebel_query_function(model, text_file: str, context_window: int, pkl_dump: str):
    rebel_output = get_rebel_output_on_text_file(text_file)
    out_taxa_data = convert_rebel_output_to_TaxaData(rebel_output)
    with open(pkl_dump, "wb") as file_:
        pickle.dump(out_taxa_data, file_)
    return out_taxa_data


def simple_rebel():
    # Need to run python -m spacy download en first.

    nlp = spacy.load("en_core_web_sm")

    nlp.add_pipe("rebel", after="senter", config={
        'device': 0,  # Number of the GPU, -1 if want to use CPU
        'model_name': 'Babelscape/rebel-large'}  # Model used, will default to 'Babelscape/rebel-large' if not given
                 )
    input_sentence = "Cinchona calisaya is a plant species used for treating malaria. Another plant, Aspidosperma pubsecens L. is used to treat fever. 'natural' remedies and cutting edge neuroscience. Thirdly, the modernisation of St. John's Wort and Heantos into industrially-produced herbal medicinal products through the 1980s and 90s has in both cases been a typically global endeavour with a curious, yet not entirely fortuitous convergence in Germany. Fourthly, while the modernisation of St. John's Wort (a single plant species) began as a search for single active ingredients to explain its efficacy, Heantos is made out of thirteen different plants with scientists questioning whether or not a single active ingredient approach would be the most appropriate, providing me with insight into one of the most salient debates in the field of herbal medicine today. And finally, analysing the ways in which herbal medicine has been problematised in both an industrialised country of the West and a developing country of the East will allow me to address the oft-invoked distinction between an exotic, almost mystical Eastern medicine and a rational, Western medicine that is doing all it can to struggle against such 'superstitions'. Even a brief look at what has been happening with herbal medicine in the past four decades or so in Vietnam and the United Kingdom provides a staggering panorama of a whole 37 complex of new regulations, toxicity tests, clinical efficacy trials, scientific research programmes and standardised production procedures, centred on the key problems of safety, quality and efficacy. Previously rejected as 'fringe medicine' or 'quackery', Vietnamese and British herbal remedies are increasingly being mobilised and regulated according to their evidence bases as their sanctioning and legitimacy becomes dependent on the rigorous safety and efficacy trials that are currently favoured in biomedicine. At the same time, however, critics of biomedical notions of safety and efficacy continue to call for a rethinking of the patient, rejecting a view of him or her as one who 'merely lodges the disease' in favour of a view of the 'whole person' where safety and efficacy are linked not just to symptom-based measures of health but also to quality of life and balance. Herbal medicine patients are not 'merely' to be treated for certain health conditions, they are also to be activated out of passive roles as recipients of healthcare, 'responsibilised' into leading healthier lifestyles, provided with a framework of meaning for understanding and coping with their illnesses, and encouraged to actively improve their personal well-being and self-appraisal by taking charge of their lives. And finally, practitioners' qualifications are increasingly being scrutinised with calls for a standardisation of competency criteria and the establishing of registers or licensing systems. Hence, it would appear that what the birth of 'alternative medicine' and 'traditional medicine' has marked is a transformation of the ways in which 'quackery' is thought about and regulated against in these two countries, as well as an inauguration of new objectivities and subjectivities, as novel configurations - or dispositifs - "

    doc = nlp(input_sentence)
    doc_list = nlp.pipe([input_sentence])
    for value, rel_dict in doc._.rel.items():
        print(f"{value}: {rel_dict}")


if __name__ == '__main__':
    simple_rebel()
