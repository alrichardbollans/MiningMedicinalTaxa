For evaluating the models, we use similar evaluation methods to other RE work, e.g. [1,2], for each relation, the models are tasked with extracting
subjects and objects i.e. triplets (subject, relation, object) and the triplet is considered correct if it matches an equivalent triplet in the human
annotations. Matching triplets is considered in two setting, 'precise' and 'approximate'. Precision, recall and F1 scores are output.

For NER evaluation, the process is the same except a single phrase is extracted rather than a triplet. The LLM models are compared to a baseline
provided by TaxoNERD [3]. We use the "en_ner_eco_biobert" model for best performance. As the model is trained on common names as well, we use the
entity linking to the GBIF backbone [4] in order to remove common names.

In contrast to much RE work, in predetermining the relationships to extract the model has to interpret the semantics of the relation rather than just
extracting relevant snippets of text to represent the relation.

Also notably, parsing structured outputs of LLMs is mostly resolved through langchain and so a significant challenge of [2] is resolved.

[1] Nayak, Tapas, and Hwee Tou Ng. "Effective modeling of encoder-decoder architecture for joint entity and relation extraction." _Proceedings of the
AAAI conference on artificial intelligence_. Vol. 34. No. 05. 2020.

[2] Wadhwa, Somin, Silvio Amir, and Byron C. Wallace. "Revisiting relation extraction in the era of large language models." _Proceedings of the
conference. Association for Computational Linguistics. Meeting_. Vol. 2023. NIH Public Access, 2023.

[3] Le Guillarme, Nicolas, and Wilfried Thuiller. "TaxoNERD: deep neural models for the recognition of taxonomic entities in the ecological and
evolutionary literature." Methods in Ecology and Evolution 13.3 (2022): 625-641.

[4] GBIF Secretariat (2023). GBIF Backbone Taxonomy. Checklist dataset https://doi.org/10.15468/39omei accessed via GBIF.org on 2024-08-20. 