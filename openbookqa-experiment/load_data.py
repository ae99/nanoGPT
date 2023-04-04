import json

def read_openbookqa(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

train = read_openbookqa('../data/openbookqa/OpenBookQA-V1-Sep2018/Data/Main/train.jsonl')
val = read_openbookqa('../data/openbookqa/OpenBookQA-V1-Sep2018/Data/Main/dev.jsonl')

import re

def preprocess(text):
    text = text.strip()
    text = text.replace("  ", " ")
    return text

def process_example(example):
    out_doc = {
        "query": preprocess(example["question"]["stem"]),
        "choices": [preprocess(choice["text"]) for choice in example["question"]["choices"]],
        "gold": ord(example["answerKey"]) - ord("A"),
    }
    return out_doc

def example_to_text(example):
    text = example["query"] + "\n"
    for i, choice in enumerate(example["choices"]):
        text += f"{chr(ord('A') + i)}. {choice}\n"
    text += f"Answer: {chr(ord('A') + example['gold'])}"
    return text