import csv
import json
import sys

csv.field_size_limit(sys.maxsize)

tsv_file = "IMDb_movies.csv"
jsonl_file = "datapositive.jsonl"

def write_to_jsonl(file_path, data):
    with open(file_path, 'a') as f:
        json.dump(data, f)
        f.write('\n')

i = 0
with open(tsv_file, 'r') as tsv_file:
    reader = csv.reader(tsv_file)
    headers = next(reader, None)
    for row in reader:
        if row[8] == 'English':
            i += 1
            data1 = {'model': 'gpt-3.5-turbo',"max_tokens": 300,
                 "messages":[{"role": "user", "content": "Write a positive review about the movie '" + row[1] + "', which is directed by " + row[9] +
                                        " in a casual way. This is a brief description of the movie: '" + row[13] + "' Your response should only contain text."}]}
            write_to_jsonl(jsonl_file, data1)
            if i > 500:
                break
print("done")
