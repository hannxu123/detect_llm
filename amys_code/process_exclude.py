import json
import pickle
from datasets import load_dataset
from dataset2 import Corpus_all
import numpy as np
import random

# human_or_chat = True, 0, human
# human_or_chat = False, 1, chat

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data

def write(all_data):
    with open('/Users/amyliu/Documents/hshsp/final/all_data_exclude.jsonl', 'w') as f:
        for data in all_data:
            json.dump(data, f)
            f.write('\n')

data_v4 = load_jsonl("/Users/amyliu/Documents/hshsp/final/all_datav4.jsonl")
data_all = load_jsonl("/Users/amyliu/Documents/hshsp/final/all_data_include.jsonl")

exclude = []

i = 0

for line in data_all:
    print(i)
    if line not in data_v4:
        exclude.append(line)
    i+= 1

print(len(exclude))
write(exclude)
