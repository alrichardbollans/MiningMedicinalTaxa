"""
Chunk Label Studio annotations into BERT-sized pieces.
Splits long docs into overlapping sentence windows that fit SciBERT's 512-token limit.

CHUNKING STRATEGY:
1. Split into sentences with nltk (preserves semantic units)
2. Group sentences until reaching token limit (we chose 450 to be conservative)
3. Overlap by N sentences (to guarantee that at least 90% of the relationships are preserved)
4. Remap entity/relation offsets to chunk coordinates ?

SOME RULES
- Only preserve entities fully within chunk boundaries (if an abbreviated names is split in two subchunks it won't be kept
- Provides tracking of lost entities/relations
- Preserve relations only if both head AND tail entities within the subchunk
- Report % of relations preserved (should be >90%)

For this script we considered task_id to be the parent id. This is basically the chunk_id in the annotation file.
Chunk_id here is used for the subchunk for SciBERT.
"""

import json
import csv
from pathlib import Path
from nltk.tokenize.punkt import PunktSentenceTokenizer
from transformers import AutoTokenizer
from SciBert.config import Config


def extract_from_label_studio(task):
    text = task["data"]["text"]
    result = task.get("annotations", [{}])[0].get("result", [])

    entities = []
    id_to_idx = {}
    for item in result:
        if item.get("type") == "labels":
            v = item["value"]
            id_to_idx[item["id"]] = len(entities)
            entities.append({"start": v["start"], "end": v["end"], "label": v["labels"][0], "text": text[v["start"]:v["end"]]})

    relations = []
    for item in result:
        if item.get("type") == "relation" and item.get("labels"):
            if item["from_id"] in id_to_idx and item["to_id"] in id_to_idx:
                relations.append({"head": id_to_idx[item["from_id"]], "tail": id_to_idx[item["to_id"]], "label": item["labels"][0]})

    return text, entities, relations


def build_sentence_windows(text, sent_tokenizer, tokenizer):
    spans = list(sent_tokenizer.span_tokenize(text))
    if not spans:
        return [(0, len(text))] if text else []

    windows, i = [], 0
    while i < len(spans):
        sents, j = [], i
        while j < len(spans):
            candidate = " ".join(sents + [text[spans[j][0]:spans[j][1]]])
            if len(tokenizer(candidate, add_special_tokens=False)["input_ids"]) > Config.MAX_CHUNK_TOKENS and sents:
                break
            sents.append(text[spans[j][0]:spans[j][1]])
            j += 1
        windows.append((spans[i][0], spans[j - 1][1]))
        if j >= len(spans):
            break
        i = max(j - Config.OVERLAP_SENTENCES, i + 1)

    return windows


def chunk_document(task_id, text, entities, relations, sent_tokenizer, tokenizer):
    chunks, lost_relations = [], []
    seen_relations = set()

    for chunk_idx, (cs, ce) in enumerate(build_sentence_windows(text, sent_tokenizer, tokenizer)):
        chunk_text = text[cs:ce]
        g2l = {}
        chunk_entities = []

        for gi, e in enumerate(entities):
            if e["start"] >= cs and e["end"] <= ce:
                g2l[gi] = len(chunk_entities)
                chunk_entities.append({"start": e["start"] - cs, "end": e["end"] - cs, "label": e["label"], "text": chunk_text[e["start"]-cs:e["end"]-cs]})

        chunk_relations = []
        for r in relations:
            if r["head"] in g2l and r["tail"] in g2l:
                chunk_relations.append({"head": g2l[r["head"]], "tail": g2l[r["tail"]], "label": r["label"]})
                seen_relations.add((r["head"], r["tail"], r["label"]))

        if chunk_entities: # this discards empty chunks
            chunks.append({"task_id": task_id, "chunk_id": chunk_idx, "text": chunk_text, "entities": chunk_entities, "relations": chunk_relations})

    for r in relations:
        key = (r["head"], r["tail"], r["label"])
        if key not in seen_relations:
            lost_relations.append({"task_id": task_id, "label": r["label"], "head": entities[r["head"]]["text"], "tail": entities[r["tail"]]["text"]})

    return chunks, lost_relations


def main():
    Config.validate()

    with open(Config.LABEL_STUDIO_FILE, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    sent_tokenizer = PunktSentenceTokenizer()
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    all_chunks, all_lost = [], []
    for task in tasks:
        text, entities, relations = extract_from_label_studio(task)
        chunks, lost = chunk_document(task["id"], text, entities, relations, sent_tokenizer, tokenizer)
        all_chunks.extend(chunks)
        all_lost.extend(lost)

    output_file = Config.OUTPUTS / "scibert_chunks.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    if all_lost:
        lost_file = Config.OUTPUTS / "lost_relations.csv"
        with open(lost_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["task_id", "label", "head", "tail"])
            writer.writeheader()
            writer.writerows(all_lost)

    print(f"Done. {len(all_chunks)} chunks, {len(all_lost)} lost relations.")


    if __name__ == "__main__":
        main()