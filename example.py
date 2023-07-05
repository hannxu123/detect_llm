
import json

def write_to_jsonl(file_path, data):
    with open(file_path, 'a') as f:
        json.dump(data, f)
        f.write('\n')

# Example data
data1 = {'name': 'John', 'age': 30, 'city': 'New York'}
data2 = {'name': 'Jane', 'age': 25, 'city': 'San Francisco'}
data3 = {'name': 'Bob', 'age': 40, 'city': 'Chicago'}

# File path
file_path = 'data.jsonl'

# Write data to the JSONL file
write_to_jsonl(file_path, data1)
write_to_jsonl(file_path, data2)
write_to_jsonl(file_path, data3)