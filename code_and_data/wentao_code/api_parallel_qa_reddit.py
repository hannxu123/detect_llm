import json
import pickle

def write_to_jsonl(file_path, data):
    with open(file_path, 'a') as f:
        json.dump(data, f)
        f.write('\n')

# Example data
file_path = './qa_reddit_p2_3000.jsonl'

with open('./cache_dir/reddit_data', 'rb') as f: 
    data = pickle.load(f)

for j in range(3000):
    if ' Please explain like I\'m five.' in data[j]:
        new_data_temp = data[j].replace(' Please explain like I\'m five.', '')
        #new_data = data[j].replace(' Please explain like I\'m five.', ' Please explain no more than 100 words like I\'m five.')
    elif 'Explain like I\'m five.' in data[j]:
        new_data_temp = data[j].replace(' Explain like I\'m five.', '')
        #new_data = data[j].replace(' Explain like I\'m five.', ' Please explain no more than 100 words like I\'m five.')
    else:
        print(f' index is {j} and data is {data[j]}')
    #new_data = 'Answer the following question no more than 100 words: ' + new_data_temp #p1
    new_data = 'Answer the following question no more than 100 words: ' + new_data_temp + ' Please explain like I\'m five.' #p2
    #new_data = 'Act as if you are a user in Reddit, answer the following question in the most simple terms, as you would to a child: ' + new_data_temp + ' Do not include user id in the answer.'#p3
    data1 = {'model': 'gpt-3.5-turbo', 'max_tokens': 500, 'messages': [{'role': 'user', 'content': new_data}]}

    # Write data to the JSONL file
    write_to_jsonl(file_path, data1)
