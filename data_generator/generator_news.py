import json
import pickle
import random
import re
from datasets import load_dataset

def write_to_jsonl(file_path, data):
    with open(file_path, 'a') as f:
        json.dump(data, f)
        f.write('\n')

jsonl_file = "business.jsonl"

data_set = load_dataset("ag_news")['train']

n = 0
for j in range(len(data_set)):

    title = data_set[j]['text']
    topic = data_set[j]['label']

    if topic == 2:
        number = random.choice(['100', '200'])
        news = random.choice(['news article', 'article'])

        new_data = 'Write a ' + number + ' words ' + news + ' following the summary: ' + title + 'Don\' show the title.'
        data1 = {'model': 'gpt-3.5-turbo', 'max_tokens': 500, 'temperature': 0.7, 'messages': [{'role': 'user', 'content': new_data}]}
        print(data1)
        write_to_jsonl(jsonl_file, data1)
        n = n + 1
        if n > 4500:
            break