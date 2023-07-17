import pickle
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
import json
from datasets import load_dataset
from datasets.utils.logging import disable_progress_bar
disable_progress_bar()
import datasets
datasets.logging.set_verbosity_error()
import os
import random
import torch

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data

def process_spaces(story):
    story = story[0].upper() + story[1:]
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

def strip_newlines(text):
    return ' '.join(text.split())

def cut_data(data):
    data = [x.strip() for x in data]
    data = [strip_newlines(x) for x in data]
    data = [process_spaces(x) for x in data]
    long_data = [x for x in data if (len(x.split()) > 32)]
    return long_data

def Corpus_HC3():

    dataset = load_dataset("Hello-SimpleAI/HC3", name='all', cache_dir='chat')
    real_list = dataset['train']['human_answers']
    fake_list = dataset['train']['chatgpt_answers']

    real_data = []
    for real in real_list:
        for sentence in real:
            real_data.append(sentence)

    fake_data = []
    for fake in fake_list:
        for sentence in fake:
            fake_data.append(sentence)

    random.shuffle(real_data)
    random.shuffle(fake_data)

    real_data = cut_data(real_data)
    fake_data = cut_data(fake_data)

    real_train = real_data[0:len(real_data) - 300]
    real_test = real_data[len(real_data) - 300:]

    fake_train = fake_data[0:len(fake_data) - 300]
    fake_test = fake_data[len(fake_data) - 300:]

    return real_train, real_test, fake_train, fake_test



def Corpus_imdb(prompt = 'p1'):

    fake_train = []
    fake_valid = []
    fake_test = []

    if (prompt == 'p1') or (prompt == 'p2') or (prompt == 'p3'):
        fake_data = []
        file_path = 'data_save/chat_imdb_' + prompt + '_results.jsonl'
        jsonl_data = load_jsonl(file_path)
        for message in jsonl_data:
            fake_data.append(message[1]['choices'][0]['message']['content'])
        fake_data = cut_data(fake_data)
        fake_train = fake_data[0:len(fake_data) - 400]
        fake_valid = fake_data[len(fake_data) - 400: len(fake_data) - 250]
        fake_test = fake_data[len(fake_data) - 250:]

    elif (prompt == 'all1'):
        for p in ['p1', 'p2', 'p3']:
            temp_data = []
            file_path = 'data_save/chat_imdb_' + p + '_results.jsonl'
            jsonl_data = load_jsonl(file_path)
            for message in jsonl_data:
                temp_data.append(message[1]['choices'][0]['message']['content'])
            temp_data = cut_data(temp_data)
            fake_train.extend(temp_data[0:900])
            fake_valid.extend(temp_data[len(temp_data) - 400: len(temp_data) - 250])
            fake_test.extend(temp_data[len(temp_data) - 250:])

    elif (prompt == 'all2'):
        for p in ['p1', 'p2', 'p3']:
            temp_data = []
            file_path = 'data_save/chat_imdb_' + p + '_results.jsonl'
            jsonl_data = load_jsonl(file_path)
            for message in jsonl_data:
                temp_data.append(message[1]['choices'][0]['message']['content'])
            temp_data = cut_data(temp_data)
            fake_train.extend(temp_data[0:len(temp_data) - 400])
            fake_valid.extend(temp_data[len(temp_data) - 400: len(temp_data) - 250])
            fake_test.extend(temp_data[len(temp_data) - 250:])
    else:
        raise ValueError

    ## processing human data
    real_dataset = load_dataset("imdb", cache_dir='./cache_dir')['train']
    real_data = []
    for i in range(10000):   ## select 10000 imdb original review as human written
        real_data.append(real_dataset[i]['text'])
    real_data = cut_data(real_data)
    real_train = real_data[0:len(real_data) - 400]
    real_valid = real_data[len(real_data) - 400: len(real_data) - 250]
    real_test = real_data[len(real_data) - 250:]

    return real_train, real_test, real_valid, fake_train, fake_test, fake_valid


def Corpus_amazon(prompt = 'p1'):

    fake_train = []
    fake_valid = []
    fake_test = []

    if (prompt == 'p1') or (prompt == 'p2') or (prompt == 'p3'):
        fake_data = []
        file_path = 'data_save/chat_amazon_' + prompt + '_results.jsonl'
        jsonl_data = load_jsonl(file_path)
        for message in jsonl_data:
            fake_data.append(message[1]['choices'][0]['message']['content'])
        fake_data = cut_data(fake_data)
        fake_train = fake_data[0:len(fake_data) - 400]
        fake_valid = fake_data[len(fake_data) - 400: len(fake_data) - 250]
        fake_test = fake_data[len(fake_data) - 250:]

    elif (prompt == 'all1'):
        for p in ['p1', 'p2', 'p3']:
            temp_data = []
            file_path = 'data_save/chat_amazon_' + p + '_results.jsonl'
            jsonl_data = load_jsonl(file_path)
            for message in jsonl_data:
                temp_data.append(message[1]['choices'][0]['message']['content'])
            temp_data = cut_data(temp_data)
            fake_train.extend(temp_data[0:900])
            fake_valid.extend(temp_data[len(temp_data) - 400: len(temp_data) - 250])
            fake_test.extend(temp_data[len(temp_data) - 250:])

    elif (prompt == 'all2'):
        for p in ['p1', 'p2', 'p3']:
            temp_data = []
            file_path = 'data_save/chat_amazon_' + p + '_results.jsonl'
            jsonl_data = load_jsonl(file_path)
            for message in jsonl_data:
                temp_data.append(message[1]['choices'][0]['message']['content'])
            temp_data = cut_data(temp_data)
            fake_train.extend(temp_data[0:len(temp_data) - 400])
            fake_valid.extend(temp_data[len(temp_data) - 400: len(temp_data) - 250])
            fake_test.extend(temp_data[len(temp_data) - 250:])
    else:
        raise ValueError

    ## processing human data
    with open('./cache_dir/amazon_review_data_human', 'rb') as f:
        real_data = pickle.load(f)

    #real_dataset = load_dataset("imdb", cache_dir='./cache_dir')['train']
    #real_data = []
    #for i in range(10000):   ## select 10000 imdb original review as human written
        #real_data.append(real_dataset[i]['text'])
    real_data = cut_data(real_data)
    real_train = real_data[0:len(real_data) - 400]
    real_valid = real_data[len(real_data) - 400: len(real_data) - 250]
    real_test = real_data[len(real_data) - 250:]

    return real_train, real_test, real_valid, fake_train, fake_test, fake_valid


def Corpus_reddit(prompt):
    fake_train = []
    fake_valid = []
    fake_test = []

    if (prompt == 'p1') or (prompt == 'p2') or (prompt == 'p3'):
        fake_data = []
        file_path = 'data_save_2/chat_reddit_' + prompt + '_results.jsonl'
        jsonl_data = load_jsonl(file_path)
        for message in jsonl_data:
            fake_data.append(message[1]['choices'][0]['message']['content'])
        fake_data = cut_data(fake_data)
        fake_train = fake_data[0:len(fake_data) - 400]
        fake_valid = fake_data[len(fake_data) - 400: len(fake_data) - 250]
        fake_test = fake_data[len(fake_data) - 250:]

    elif (prompt == 'all1'):
        for p in ['p1', 'p2', 'p3']:
            temp_data = []
            file_path = 'data_save_2/chat_reddit_' + p + '_results.jsonl'
            jsonl_data = load_jsonl(file_path)
            for message in jsonl_data:
                temp_data.append(message[1]['choices'][0]['message']['content'])
            temp_data = cut_data(temp_data)
            fake_train.extend(temp_data[0:900])
            fake_valid.extend(temp_data[len(temp_data) - 400: len(temp_data) - 250])
            fake_test.extend(temp_data[len(temp_data) - 250:])

    elif (prompt == 'all2'):
        for p in ['p1', 'p2', 'p3']:
            temp_data = []
            file_path = 'data_save_2/chat_reddit_' + p + '_results.jsonl'
            jsonl_data = load_jsonl(file_path)
            for message in jsonl_data:
                temp_data.append(message[1]['choices'][0]['message']['content'])
            temp_data = cut_data(temp_data)
            fake_train.extend(temp_data[0:len(temp_data) - 400])
            fake_valid.extend(temp_data[len(temp_data) - 400: len(temp_data) - 250])
            fake_test.extend(temp_data[len(temp_data) - 250:])
    else:
        raise ValueError

    ## processing human data
    with open('./cache_dir/reddit_data_selected_human', 'rb') as f:
        real_data = pickle.load(f)
        
    #real_data = []
    #import ipdb
    #ipdb.set_trace()
    #for i in range(10000):   ## select 10000 imdb original review as human written
        #if len(human_data[i]) == 1:
            #real_data.append(human_data[i][0])
        #else:
            #real_data.append(random.choice(human_data[i]))
    #import ipdb
    #ipdb.set_trace()
    real_data = cut_data(real_data)
    real_train = real_data[0:len(real_data) - 400]
    real_valid = real_data[len(real_data) - 400: len(real_data) - 250]
    real_test = real_data[len(real_data) - 250:]
    #import ipdb
    #ipdb.set_trace()

    return real_train, real_test, real_valid, fake_train, fake_test, fake_valid


class TextDataset(Dataset):
    def __init__(self, real_texts, fake_texts):
        self.real_texts = real_texts
        self.fake_texts = fake_texts

    def __len__(self):
        return len(self.real_texts) + len(self.fake_texts)

    def __getitem__(self, index):
        if index < len(self.real_texts):
            answer = self.real_texts[index]
            label = 0
        else:
            answer = self.fake_texts[index - len(self.real_texts)]
            label = 1
        return answer, label

def loader(batch_size, name = 'HC3', prompt = 'all2', verbose = False):
    if name == 'HC3':
        real_train, real_test, real_valid, fake_train, fake_test, fake_valid  = Corpus_HC3()
    elif name == 'imdb':
        print(f'load {name} dataset')
        real_train, real_test, real_valid, fake_train, fake_test, fake_valid = Corpus_imdb(prompt)
    elif name == 'amazon':
        print(f'load {name} dataset')
        real_train, real_test, real_valid, fake_train, fake_test, fake_valid = Corpus_amazon(prompt)
    elif name == 'reddit':
        print(f'load {name} dataset')
        real_train, real_test, real_valid, fake_train, fake_test, fake_valid = Corpus_reddit(prompt)
    else:
        raise ValueError

    if verbose:
        print('Total human texts ' + str(len(real_train)) + ', Total generated texts ' + str(len(fake_train)))
    weight = torch.cat([len(fake_train) / len(real_train) * torch.ones(len(real_train)), torch.ones(len(fake_train))])
    train_sampler = WeightedRandomSampler(weight, len(real_train) + len(fake_train), replacement=False)

    train_dataset = TextDataset(real_train, fake_train)
    train_loader = DataLoader(train_dataset, batch_size, sampler=train_sampler, num_workers=0)

    test_dataset = TextDataset(real_test, fake_test)
    test_loader = DataLoader(test_dataset, 1, shuffle= True, num_workers=0)

    valid_dataset = TextDataset(real_valid, fake_valid)
    valid_loader = DataLoader(valid_dataset, 1, shuffle= True, num_workers=0)

    return train_loader, test_loader, valid_loader
