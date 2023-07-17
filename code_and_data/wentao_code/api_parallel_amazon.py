import json
import pickle
import random
import re

# data[i]['review_body']

def write_to_jsonl(file_path, data):
    with open(file_path, 'a') as f:
        json.dump(data, f)
        f.write('\n')

# Example data
file_path = './amazon_p3_3000.jsonl'

with open('./cache_dir/amazon_review_data', 'rb') as f: 
    app_data = pickle.load(f)
    ele_data = pickle.load(f)

data = []
data.extend(app_data[:1500])
data.extend(ele_data[:1500])
for j in range(3000):
    # p1
    num_words = random.choice([50, 100])
    #new_data = 'Write a review around ' + str(num_words) + ' words for the product \"' + data[j]['product_title'] + '\" that generated the following review headline: ' + data[j]['review_headline']
    
    # p2
    #new_data = 'Complete the following product review around ' + str(num_words) + ' words for \"' + data[j]['product_title'] + '\": ' + data[j]['review_headline'] + '. Your review should contain only the review text, not the headline, and be natural.'
    
    # p3
    new_data = 'Finish the following review for the product \"' + data[j]['product_title'] + '\" around ' + str(num_words) + ' words with the following review headline: ' + data[j]['review_headline'] + '. Your review should only contain the review text, not the headline, and start like so: ' + ' '.join(re.split(r'(?<=[.?!])\s', data[j]['review_body'])[:1])
    
    data1 = {'model': 'gpt-3.5-turbo', 'max_tokens': 500, 'messages': [{'role': 'user', 'content': new_data}]}

    # Write data to the JSONL file
    write_to_jsonl(file_path, data1)
