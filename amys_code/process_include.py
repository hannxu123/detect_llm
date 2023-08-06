import json
import pickle
from datasets import load_dataset
from dataset2 import Corpus_all
import numpy as np
import random

# human_or_chat = True, 0, human
# human_or_chat = False, 1, chat


all_data = []
prev = 0

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data

def loadFromDataset(name, task):
    global all_data
    if( name == 'Ivypanda'):
        dataset = load_dataset("qwedsacf/ivypanda-essays")['train']
        t = 'TEXT'
    elif( name == 'IMDb'):
        dataset = load_dataset("imdb")['train']
        t = 'text'
    for i in range(10000):
        text = dataset[i][t]
        label = 0
        name = name
        task = task
        data = {'text': text, 'label': label, 'prompt': 'None', 'name': name, 'task': task}
        all_data.append(data)

def printInfo( dataset, coh, length):
    global prev
    # print( dataset + " " + coh + ": " + (str(all_data[-1]['text'])))
    print( dataset + " " + coh + " length: " + str(length - prev))
    print("------------------")
    prev = len(all_data)


#copies all info from a json file, no prompt, for chatgpt samples
def run(file, name, task):
    global all_data
    file = "/Users/amyliu/Documents/hshsp/final/data_save/" + file
    jsonl_data = load_jsonl(file)

    for i in range(len(jsonl_data)-1):
        line = jsonl_data[i]
        label = 1
        text = line[1]['choices'][0]['message']['content']
        data = {'text': text, 'label': label, 'prompt': 'None', 'name': name, 'task': task}

        all_data.append(data)

#copies info in random_idx into all_data, prompt exists, for chatgpt samples
def runRandIdx( file, name, task, prompt):
    global all_data
    file = "/Users/amyliu/Documents/hshsp/final/data_save/" + file
    jsonl_data = load_jsonl(file)

    for i in range(len(jsonl_data)):
        label = 1
        text = jsonl_data[i][1]['choices'][0]['message']['content']
        data = {'text': text, 'label': label, 'prompt': prompt, 'name': name, 'task': task}

        all_data.append(data)


def runHuman(file, name, task):
    global all_data
    fileload = "/Users/amyliu/Documents/hshsp/final/data_save/" + file
    with open(fileload, "rb") as fp:
        jsonl_data = pickle.load(fp)
        if( file == 'real_news'):
            jsonl_data = jsonl_data[name.lower()]

    for i in range(len(jsonl_data)):
        label = 0
        text = jsonl_data[i]
        if( type(text) == list):
            text = ' '.join(jsonl_data[i])
        if( text == ""):
            continue

        data = {'text': text, 'label': label, 'prompt': 'None', 'name': name, 'task': task}

        all_data.append(data)


def write(all_data):
    with open('/Users/amyliu/Documents/hshsp/final/all_data_include.jsonl', 'w') as f:
        for data in all_data:
            json.dump(data, f)
            f.write('\n')

eli_files = ['eli5_p1_results.jsonl', 'eli5_p2_results.jsonl', 'eli5_p3_results.jsonl']
imdb_files = ['imdb_p1_results.jsonl', 'imdb_p2_results.jsonl', 'imdb_p3_results.jsonl']
amazon_files = ['amazon_p1_results.jsonl', 'amazon_p2_results.jsonl', 'amazon_p3_results.jsonl']
askh_files = ['askh_p1_results.jsonl', 'askh_p2_results.jsonl', 'askh_p3_results.jsonl']
asks_files = ['asks_p1_results.jsonl', 'asks_p2_results.jsonl', 'asks_p3_results.jsonl']


##NEWS
run('world_results.jsonl', 'World', 'News')
printInfo( "world", "chat", len(all_data))

runHuman('real_news', 'World', 'News')
printInfo( "world", "human", len(all_data))



run('sports_results.jsonl', 'Sports', 'News')
printInfo( "sports", "chat", len(all_data))

runHuman('real_news', 'Sports', 'News')
printInfo( "sports", "human", len(all_data))



run('business_results.jsonl', 'Business', 'News')
printInfo( "business", "chat", len(all_data))

runHuman('real_news', 'Business', 'News')
printInfo( "business", "human", len(all_data))



##IMDB
n = 1
for file in imdb_files:
    runRandIdx(file, 'IMDb', 'Review', n )
    printInfo( "imdb", "chat", len(all_data))
    n += 1

loadFromDataset('IMDb', 'Review')
printInfo( "imdb", "human", len(all_data))


##AMAZON
n = 1
for file in amazon_files:
    runRandIdx(file, 'Amazon', 'Review', n )
    printInfo( "amazon", "chat", len(all_data))
    n += 1


runHuman('real_amazon', 'Amazon', 'Review')
printInfo( "amazon", "human", len(all_data))



##IVYPANDA
run('ivy_results.jsonl', 'Ivypanda', 'Writing')
printInfo( "ivypanda", "chat", len(all_data))

loadFromDataset('Ivypanda', 'Writing')
printInfo( "ivypanda", "human", len(all_data))




##QA
n = 1
for file in eli_files:
    runRandIdx(file, 'Eli5', 'QA', n)
    n += 1
printInfo( "eli5", "chat", len(all_data))

runHuman('real_eli5', 'Eli5', 'QA')
printInfo( "eli5", "human", len(all_data))

for file in askh_files:
    runRandIdx(file, 'AskHist.', 'QA', 'None')
printInfo( "askh", "chat", len(all_data))

runHuman('real_askh', 'AskHist.', 'QA')
printInfo( "askh", "human", len(all_data))


for file in asks_files:
    runRandIdx(file, 'AskSci.', 'QA', 'None')
printInfo( "asks", "chat", len(all_data))

runHuman('real_asks', 'AskSci.', 'QA')
printInfo( "asks", "human", len(all_data))

write(all_data)
