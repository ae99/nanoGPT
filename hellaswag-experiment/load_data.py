import json

def read_hellaswag(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

train = read_hellaswag('../data/hellaswag/hellaswag_train.jsonl')
val = read_hellaswag('../data/hellaswag/hellaswag_val.jsonl')

import re

def preprocess(text):
    text = text.strip()
    # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text

def process_example(example):
    ctx = example["ctx_a"] + " " + example["ctx_b"].capitalize()
    out_doc = {
        "query": preprocess(example["activity_label"] + ": " + ctx),
        "choices": [preprocess(ending) for ending in example["endings"]],
        "gold": int(example["label"]),
    }
    return out_doc

def example_to_text(example):
    text = example["query"] + "\n"
    for i, choice in enumerate(example["choices"]):
        text += f"{i+1}. {choice}\n"
    text += f"Answer: {example['gold'] + 1}"   
    return text
