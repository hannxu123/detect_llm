
import json

def write_to_jsonl(file_path, data):
    with open(file_path, 'a') as f:
        json.dump(data, f)
        f.write('\n')

# Example data
file_path = 'data.jsonl'

for j in range(5):
    data1 = {'model': 'gpt-3.5-turbo', 'max_tokens': 30, 'messages': [{'role': 'user', 'content': 'What is 1 + ' + str(j)}]}

    # Write data to the JSONL file
    write_to_jsonl(file_path, data1)