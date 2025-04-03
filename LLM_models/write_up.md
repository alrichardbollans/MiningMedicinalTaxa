For evaluating the models, we use similar evaluation methods to other RE work, e.g. [1,2], for each relation, the models are tasked with extracting
subjects and objects i.e. triplets (subject, relation, object) and the triplet is considered correct if it matches an equivalent triplet in the human
annotations. Matching triplets is considered in two setting, 'precise' and 'approximate'. Precision, recall and F1 scores are output.

For NER evaluation, the process is the same except a single phrase is extracted rather than a triplet. The LLM models are compared to a baseline
provided by TaxoNERD [3]. We use the "en_ner_eco_biobert" model for best performance. As the model is trained on common names as well, we use
taxonomic knowledge to remove common names.

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


### Costs
For a single hparam run (01/04/25):
- Claude - \$0.59
- LLama via Fireworks - \$0.49
- gpt via openai - \$0.45
- deepseek via deepseek api - \$0.07


For evaluation (approx 9 papers), these were the following incurred costs:
- Claude - \$ 5.14
- LLama via Fireworks - \$5.03
- gpt via openai - \$3.59
- deepseek via deepseek api - \$0.22

Note that these are only approximate and extra costs may have been incurred from retrying chunks by splitting multiple times.

For finetuning gpt4o - \$ 12.06?
("hyperparameters": {
    "n_epochs": 4,
    "batch_size": 1,
    "learning_rate_multiplier": 2
  }:)
Evaluation of gpt4o_FT - \$5.33?

104 is a big chunk!
