import csv
import json
import random
import sys
from datasets import load_dataset

dataset = load_dataset("imdb")['train']
csv.field_size_limit(sys.maxsize)

tsv_file = "IMDb_movies.csv"
jsonl_file = "imdb_review2.jsonl"

def write_to_jsonl(file_path, data):
    with open(file_path, 'a') as f:
        json.dump(data, f)
        f.write('\n')


def process_spaces(story):
    return story.replace(
        ' ,', ',').replace(
        ' .', '.').replace(
        ' ?', '?').replace(
        ' !', '!').replace(
        ' ;', ';').replace(
        ' \'', '\'').replace(
        ' ’ ', '\'').replace(
        ' :', ':').replace(
        '<newline>', '\n').replace(
        '`` ', '"').replace(
        ' \'\'', '"').replace(
        '\'\'', '"').replace(
        '.. ', '... ').replace(
        ' )', ')').replace(
        '( ', '(').replace(
        ' n\'t', 'n\'t').replace(
        ' i ', ' I ').replace(
        ' i\'', ' I\'').replace(
        '\\\'', '\'').replace(
        '\n ', '\n').replace(
        '<br />', '').strip()

i = 0
with open(tsv_file, 'r') as tsv_file:
    reader = csv.reader(tsv_file)
    headers = next(reader, None)
    for row in reader:
        if row[8] == 'English':
            i += 1
            for j in range(1000):
                example_review = random.choice(dataset)
                if len(example_review['text'].split()) < 200:
                    break

            sentiment = 'worse. The reason I hate it is: ' if example_review['label'] == 1 else 'better. The reason I love it is: '
            data1 = {'model': 'gpt-3.5-turbo',"max_tokens": 200,
                 "messages":[{"role": "user", "content":
                           'Complete the following: ' + process_spaces(example_review['text']) +
                             " But \"" + row[1] + "\" is much " + sentiment
                              }]}
            print(data1)
            write_to_jsonl(jsonl_file, data1)
            if i > 10:
                break
print("done")
