import json
import random

import pandas as pd

df1 = pd.read_csv('data/imdb/title.akas.tsv.gz', compression='gzip', delimiter='\t')
df1 = df1[df1['language'] == 'en']  ## only english
df1.rename(columns={'titleId': 'tconst'}, inplace=True)
df1 = df1.dropna(subset=['title'])  ## remove the records without a title
df1 = df1.drop_duplicates(subset='tconst')

df2 = pd.read_csv('data/imdb/title.ratings.tsv.gz', compression='gzip', delimiter='\t')
df2 = df2[df2['numVotes'] > 15000]

df = pd.merge(df2, df1, on='tconst', how='left')
df = df.dropna(subset=['title'])  ## remove the records without a title
df['title'] = df['title'].astype('str')

jsonl_file = "data/imdb/chat_imdb_p3.jsonl"

def write_to_jsonl(file_path, data):
    with open(file_path, 'a') as f:
        json.dump(data, f)
        f.write('\n')

titles = list(df['title'])
for i in range(3000):
    print(titles[i])
    sense = random.choice(['enjoyable', 'just OK', 'mediocre', 'unpleasant', 'great'])
    prompt = random.choice(['It is because that', 'The reason is that', 'I just feel that ', 'I am feeling that '])

    data1 = {'model': 'gpt-3.5-turbo',"max_tokens": 200,
             "messages":[{"role": "user", "content":
             "Complete the following: I just watched \"" + titles[i] + "\". It is " + sense + '. ' + prompt}]}
    print(data1)
    write_to_jsonl(jsonl_file, data1)

print('Total number of movies ' +str(len(df)), 'We have collected for 3000')
print("done")
