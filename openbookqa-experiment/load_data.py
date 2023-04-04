import json

def read_gsm8k(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

train = read_gsm8k('../data/gsm8k/train.jsonl')
val = read_gsm8k('../data/gsm8k/test.jsonl')


def example_to_text(example):
    return example['question'] + '\n' + example['answer'] + '<|endoftext|>'

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"
