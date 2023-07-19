import json
import pickle
import random
import re
from datasets import load_dataset

def write_to_jsonl(file_path, data):
    with open(file_path, 'a') as f:
        json.dump(data, f)
        f.write('\n')

jsonl_file = "ivy.jsonl"
data = load_dataset("qwedsacf/ivypanda-essays")['train']


for j in range(4500):
    number = random.choice(['300', '400'])
    style = random.choice(['a high school student', 'a child', 'a novelist'])

    title = data[j]['TEXT'].split('\n')[0]
    new_data = 'Write a ' + number + ' words essay like ' + style +' with the following title: ' + title + '. Your response should only contain the writing essay body.'
    data1 = {'model': 'gpt-3.5-turbo', 'max_tokens': 500, 'temperature': 0.7, 'messages': [{'role': 'user', 'content': new_data}]}
    print(data1)

    # Write data to the JSONL file
    write_to_jsonl(jsonl_file, data1)

