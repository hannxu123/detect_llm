import json
import pickle
import random
import re
from datasets import load_dataset

def write_to_jsonl(file_path, data):
    with open(file_path, 'a') as f:
        json.dump(data, f)
        f.write('\n')

jsonl_file = "asks_p1.jsonl"
data = load_dataset("eli5")['train_asks']

for j in range(1500):
    question = data[j]['title']
    number = random.choice(['50', '100', '150'])

    new_data = 'Answer the following question no more than ' + number +' words: ' + question
    # new_data = 'Answer the following question no more than ' + number +' words: '  + question + ' Please explain like I\'m five.' #p2
    # new_data = 'Act as if you are a user in Reddit, answer the following question in the most simple terms, as you would to a child: ' \
    #           + question + ' Do not include user id in the answer.'#p3

    data1 = {'model': 'gpt-3.5-turbo', 'max_tokens': 500, 'temperature': 0.7, 'messages': [{'role': 'user', 'content': new_data}]}
    print(data1)

    # Write data to the JSONL file
    write_to_jsonl(jsonl_file, data1)

