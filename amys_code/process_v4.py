import json
import pickle
from datasets import load_dataset
from dataset2 import Corpus_all
import numpy as np
import random

# human_or_chat = True, 0, human
# human_or_chat = False, 1, chat

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data

def loadFromDataset(name, task):
    all_data = []
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
    return all_data


all_data = []
prev = 0

def printInfo( dataset, coh, length):
    global prev
    # print( dataset + " " + coh + ": " + (str(all_data[-1]['text'])))
    print( dataset + " " + coh + " length: " + str(length - prev))
    print("------------------")
    prev = len(all_data)


#copies all info from a json file, no prompt, for chatgpt samples
def run(file, name, task):
    file = "data_save/" + file
    jsonl_data = load_jsonl(file)

    for i in range(4500):
        line = jsonl_data[i]
        label = 1
        text = line[1]['choices'][0]['message']['content']
        data = {'text': text, 'label': label, 'prompt': 'None', 'name': name, 'task': task}

        all_data.append(data)

#copies info in random_idx into all_data, prompt exists, for chatgpt samples
def runRandIdx( name, task, random_idx, prompt):
    file = "data_save/" + file
    jsonl_data = load_jsonl(file)

    for i in range(len(jsonl_data)):
        if( i in random_idx):
            label = 1
            text = jsonl_data[i][1]['choices'][0]['message']['content']
            data = {'text': text, 'label': label, 'prompt': prompt, 'name': name, 'task': task}

            all_data.append(data)


def runHuman(file, name, task):
    all_data = []
    fileload = "data_save/" + file
    with open(fileload, "rb") as fp:
        if( file == 'real_news'):
            jsonl_data = jsonl_data[name.lower()]
        else:
            jsonl_data = pickle.load(fp)

    n = 0
    for i in range(len(jsonl_data)):
        label = 0
        n += 1
        text = jsonl_data[i]
        if( type(text) == list):
            text = ' '.join(jsonl_data[i])
        if( text == ""):
            n -= 1
            continue

        data = {'text': text, 'label': label, 'prompt': 'None', 'name': name, 'task': task}

        all_data.append(data)

        if( n > 9999):
            break


def write(all_data):
    with open('all_datav4.jsonl', 'w') as f:
        for data in all_data:
            json.dump(data, f)
            f.write('\n')

eli_files = ['eli5_p1_results.jsonl', 'eli5_p2_results.jsonl', 'eli5_p3_results.jsonl']
imdb_files = ['imdb_p1_results.jsonl', 'imdb_p2_results.jsonl', 'imdb_p3_results.jsonl']
amazon_files = ['amazon_p1_results.jsonl', 'amazon_p2_results.jsonl', 'amazon_p3_results.jsonl']
askh_files = ['askh_p1_results.jsonl', 'askh_p2_results.jsonl', 'askh_p3_results.jsonl']
asks_files = ['asks_p1_results.jsonl', 'asks_p2_results.jsonl', 'asks_p3_results.jsonl']



##NEWS
all_data.extend(run('world_results.jsonl', 'World', 'News'))
printInfo( "world", "chat", len(all_data))

all_data.extend( runHuman('real_news', 'World', 'News')) 
printInfo( "world", "human", len(all_data))



all_data.extend(run('sports_results.jsonl', 'Sports', 'News'))
printInfo( "sports", "chat", len(all_data))

all_data.extend( runHuman('real_news', 'Sports', 'News'))
printInfo( "sports", "human", len(all_data))



all_data.extend(run('business_results.jsonl', 'Business', 'News'))
printInfo( "business", "chat", len(all_data))

all_data.extend( runHuman('real_news', 'Business', 'News'))
printInfo( "business", "human", len(all_data))



##IMDB
n = 1
for file in imdb_files:
    random_idx = np.random.choice(3000,1500, replace = False)
    all_data.extend(runRandIdx(file, 'IMDb', 'Review', random_idx, n ))
    printInfo( "imdb", "chat", len(all_data))
    n += 1

all_data.extend( loadFromDataset('IMDb', 'Review'))
printInfo( "imdb", "human", len(all_data))


##AMAZON
n = 1
for file in amazon_files:
    if( file == 'amazon_p1_results.jsonl'):
        array1 = np.arange(0, 2191)
        array2 = np.arange(2195, 3001)

        random_idx = np.concatenate((array1, array2))
        random_idx = np.random.choice(random_idx,1500, replace = False)

    else:
        random_idx = np.random.choice(3000,1500, replace = False)
    all_data.extend(runRandIdx(file, 'Amazon', 'Review', random_idx, n ))
    printInfo( "amazon", "chat", len(all_data))
    n += 1


all_data.extend( runHuman('real_amazon', 'Amazon', 'Review'))
printInfo( "amazon", "human", len(all_data))



##IVYPANDA
all_data.extend(run('ivy_results.jsonl', 'Ivypanda', 'Writing'))
printInfo( "ivypanda", "chat", len(all_data))

all_data.extend( loadFromDataset('Ivypanda', 'Writing'))
printInfo( "ivypanda", "human", len(all_data))




##QA
n = 1
for file in eli_files:
    random_idx = np.random.choice(3000,1500, replace = False)
    all_data.extend(runRandIdx(file, 'Eli5', 'QA', random_idx, n))
    n += 1
printInfo( "eli5", "chat", len(all_data))

all_data.extend( runHuman('real_eli5', 'Eli5', 'QA'))
printInfo( "eli5", "human", len(all_data))

for file in askh_files:
    random_idx = np.random.choice(1500,1500, replace = False)
    all_data.extend(runRandIdx(file, 'AskHist.', 'QA', random_idx, 'None'))
printInfo( "askh", "chat", len(all_data))

all_data.extend( runHuman('real_askh', 'AskHist.', 'QA'))
printInfo( "askh", "human", len(all_data))


for file in asks_files:
    random_idx = np.random.choice(1500,1500, replace = False)
    all_data.extend(runRandIdx(file, 'AskSci.', 'QA', random_idx, 'None'))
printInfo( "asks", "chat", len(all_data))

all_data.extend( runHuman('real_asks', 'AskSci.', 'QA'))
printInfo( "asks", "human", len(all_data))

write(all_data)
