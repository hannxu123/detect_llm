import json
import pickle
import random
import re
from datasets import load_dataset

def write_to_jsonl(file_path, data):
    with open(file_path, 'a') as f:
        json.dump(data, f)
        f.write('\n')

jsonl_file = "amazon_p3.jsonl"


dataset1 = load_dataset("amazon_us_reviews", 'Apparel_v1_00')['train']
dataset2 = load_dataset("amazon_us_reviews", 'Electronics_v1_00')['train']
dataset3 = load_dataset("amazon_us_reviews", 'Software_v1_00')['train']
dataset4 = load_dataset("amazon_us_reviews", 'Sports_v1_00')['train']
dataset5 = load_dataset("amazon_us_reviews", 'Toys_v1_00')['train']
all_dataset = [dataset1, dataset2, dataset3, dataset4, dataset5]

n = 0
for dataset in all_dataset:
    for i in range(600):
        data = dataset[i]
        #length = 500
        #num_words = random.choice([50, 100])
        #new_data = 'Write a review for the product \"' + data['product_title'] + '\" in around ' + str(num_words) + ' words'
        #

        #length = 500
        #num_words = random.choice([50, 100])
        # sense = random.choice(['positive', 'negative'])
        # new_data = "Write a " + sense + " product review for \"" + data['product_title'] + "\" in " + str(num_words) + " words. " \
        #                         "Follow the writing style of Amazon product reviews. Just give me the review body."

        sense = random.choice(['amazing', 'just OK', 'unuseful', 'unpleasant', 'great'])
        length = random.choice(range(50, 150))
        prompt = random.choice(
            ['It is because that ', 'The reason is that ', 'I just feel that ', 'I am feeling that '])
        new_data = "Continue the following: I just bought \"" + data['product_title'] + "\" from Amazon. It is " + sense + '. ' + prompt

        data1 = {'model': 'gpt-3.5-turbo', 'max_tokens': length, 'temperature': 0.7, 'messages': [{'role': 'user', 'content': new_data}]}
        # Write data to the JSONL file
        print(data1)
        write_to_jsonl(jsonl_file, data1)
        n = n+1

print('Done, we have ' +str(n) + ' samples!')