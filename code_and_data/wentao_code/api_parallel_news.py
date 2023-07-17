import json
import pickle

def write_to_jsonl(file_path, data):
    with open(file_path, 'a') as f:
        json.dump(data, f)
        f.write('\n')

# Example data
file_path = './agnews_tech_3000.jsonl'

with open('./cache_dir/agnews_filtered_tech', 'rb') as f: 
    data = pickle.load(f)

for j in range(3000):
    data1 = {'model': 'gpt-3.5-turbo', 'max_tokens': 1000, 'messages': [{'role': 'user', 'content': 'Write an article no more than 400 words following summary: ' + data[j]}]}

    # Write data to the JSONL file
    write_to_jsonl(file_path, data1)
