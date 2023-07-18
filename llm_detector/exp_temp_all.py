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
import re


def remove_last_sentence(paragraph):
    # Split the paragraph into sentences
    sentences = paragraph.split('. ')
    # Check if the last sentence ends with '.', '?', or '!'
    last_sentence = sentences[-1]
    if last_sentence.endswith(('.', '?', '!')):
        return paragraph  # Return the original paragraph if last sentence is not ended by '.', '?', or '!'
    else:
        if len(sentences) > 1:
            sentences.pop()
        # Join the remaining sentences
        modified_paragraph = '. '.join(sentences) +'.'
        return modified_paragraph



def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data

def process_spaces(story):
    story = story[0].upper() + story[1:]
    story =  story.replace(
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
    story = remove_last_sentence(story)
    return story

def strip_newlines(text):
    return ' '.join(text.split())


def cut_data(data):
    data = [x.strip() for x in data]
    data = [strip_newlines(x) for x in data]
    data = [process_spaces(x) for x in data]
    long_data = [x for x in data if (len(x.split()) > 32)]
    return long_data


def Corpus_imdb(prompt='all1'):
    fake_train = []
    fake_valid = []
    fake_test = []

    if (prompt == 'p1') or (prompt == 'p2') or (prompt == 'p3'):
        fake_data = []
        file_path = './exp_temp_data/imdb_' + prompt + '_results.jsonl'
        jsonl_data = load_jsonl(file_path)
        for message in jsonl_data:
            fake_data.append(message[1]['choices'][0]['message']['content'])
        fake_data = cut_data(fake_data)

        fake_train = fake_data[0:len(fake_data) - 500]
        fake_valid = fake_data[len(fake_data) - 500: len(fake_data) - 250]
        fake_test = fake_data[len(fake_data) - 250:]

    elif (prompt == 'all1'):
        for p in ['p1', 'p2', 'p3']:
            temp_data = []
            file_path = 'exp_temp_data/imdb_' + p + '_results.jsonl'
            jsonl_data = load_jsonl(file_path)
            for message in jsonl_data:
                temp_data.append(message[1]['choices'][0]['message']['content'])
            temp_data = cut_data(temp_data)
            fake_train.extend(temp_data[0:900])
            fake_valid.extend(temp_data[len(temp_data) - 500: len(temp_data) - 250])
            fake_test.extend(temp_data[len(temp_data) - 250:])

    elif (prompt == 'all2'):
        for p in ['p1', 'p2', 'p3']:
            temp_data = []
            file_path = 'exp_temp_data/imdb_' + p + '_results.jsonl'
            jsonl_data = load_jsonl(file_path)
            for message in jsonl_data:
                temp_data.append(message[1]['choices'][0]['message']['content'])
            temp_data = cut_data(temp_data)
            fake_train.extend(temp_data[0:len(temp_data) - 500])
            fake_valid.extend(temp_data[len(temp_data) - 500: len(temp_data) - 250])
            fake_test.extend(temp_data[len(temp_data) - 250:])
    else:
        raise ValueError

    ## processing human data
    human_data_path = './exp_temp_data/imdb_human_data'
    if os.path.exists(human_data_path):
        with open(human_data_path, 'rb') as f:
            real_data = pickle.load(f)
    else:
        real_dataset = load_dataset("imdb", cache_dir='./cache_dir')['train']
        real_data = []
        for i in range(10000):   ## select 10000 imdb original review as human written
            real_data.append(real_dataset[i]['text'])

        with open(human_data_path, 'wb') as f:
            pickle.dump(real_data, f)

    real_data = cut_data(real_data)
    real_train = real_data[0:len(real_data) - 500]
    real_valid = real_data[len(real_data) - 500: len(real_data) - 250]
    real_test = real_data[len(real_data) - 250:]

    return real_train, real_test, real_valid, fake_train, fake_test, fake_valid


def Corpus_eli5(prompt='all1'):
    fake_train = []
    fake_valid = []
    fake_test = []

    if (prompt == 'p1') or (prompt == 'p2') or (prompt == 'p3'):
        fake_data = []
        file_path = './exp_temp_data/ELI5/eli5_' + prompt + '_results.jsonl'
        jsonl_data = load_jsonl(file_path)
        for message in jsonl_data:
            fake_data.append(message[1]['choices'][0]['message']['content'])
        fake_data = cut_data(fake_data)

        fake_train = fake_data[0:len(fake_data) - 500]
        fake_valid = fake_data[len(fake_data) - 500: len(fake_data) - 300]
        fake_test = fake_data[len(fake_data) - 300:]

    elif (prompt == 'all1'):  # mix 3 prompts to from 4000 + 200 + 300 generated data
        for p in ['p1', 'p2', 'p3']:
            temp_data = []
            file_path = 'exp_temp_data/ELI5/eli5_' + p + '_results.jsonl'
            jsonl_data = load_jsonl(file_path)
            for message in jsonl_data:
                temp_data.append(message[1]['choices'][0]['message']['content'])
            temp_data = cut_data(temp_data)

            if p in ['p1', 'p2']:
                fake_train.extend(temp_data[0:1300])
                fake_valid.extend(temp_data[len(temp_data) - 165: len(temp_data) - 100])
                fake_test.extend(temp_data[len(temp_data) - 100:])
            else:
                fake_train.extend(temp_data[0:1400])
                fake_valid.extend(temp_data[len(temp_data) - 170: len(temp_data) - 100])
                fake_test.extend(temp_data[len(temp_data) - 100:])
            print(f'fake train length is {len(fake_train)}')
            print(f'fake valid length is {len(fake_valid)}')
            print(f'fake test length is {len(fake_test)}')

    elif (prompt == 'all2'):
        for p in ['p1', 'p2', 'p3']:
            temp_data = []
            file_path = 'exp_temp_data/ELI5/eli5_' + p + '_results.jsonl'
            jsonl_data = load_jsonl(file_path)
            for message in jsonl_data:
                temp_data.append(message[1]['choices'][0]['message']['content'])
            temp_data = cut_data(temp_data)
            fake_train.extend(temp_data[0:len(temp_data) - 500])
            fake_valid.extend(temp_data[len(temp_data) - 500: len(temp_data) - 300])
            fake_test.extend(temp_data[len(temp_data) - 300:])
    else:
        raise ValueError

    ## processing human data
    with open('./exp_temp_data/ELI5/real_eli5', 'rb') as f:
        real_data = pickle.load(f)
    real_data = cut_data(real_data)
    real_train = real_data[0:len(real_data) - 500]
    real_valid = real_data[len(real_data) - 500: len(real_data) - 300]
    real_test = real_data[len(real_data) - 300:]

    print(f'real train length is {len(real_train)}')
    print(f'real valid length is {len(real_valid)}')
    print(f'real test length is {len(real_test)}')

    return real_train, real_test, real_valid, fake_train, fake_test, fake_valid


class TextDataset(Dataset):
    def __init__(self, real_texts, fake_texts, tokenizer, max_sequence_length):
        self.real_texts = real_texts
        self.fake_texts = fake_texts
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length

    def __len__(self):
        return len(self.real_texts) + len(self.fake_texts)

    def __getitem__(self, index):
        if index < len(self.real_texts):
            text = self.real_texts[index]
            label = 0
        else:
            text = self.fake_texts[index - len(self.real_texts)]
            label = 1

        tokenized_test = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_sequence_length, return_tensors='pt')
        return tokenized_test['input_ids'][0], tokenized_test['attention_mask'][0], label


def exp_loader(tokenizer, batch_size, name='IMDB', prompt='all1', padding_max_length=256, train=True):
    if name == 'IMDB':
        real_train, real_test, real_valid, fake_train, fake_test, fake_valid = Corpus_imdb(prompt)
    if name == 'ELI5':
        real_train, real_test, real_valid, fake_train, fake_test, fake_valid = Corpus_eli5(prompt)
    else:
        raise ValueError

    print('Total training human texts ' + str(len(real_train)) + ', Total generated texts ' + str(len(fake_train)))

    if train:
        weight = torch.cat([len(fake_train) / len(real_train) * torch.ones(len(real_train)), torch.ones(len(fake_train))])
        train_sampler = WeightedRandomSampler(weight, len(real_train) + len(fake_train), replacement=False)

        train_dataset = TextDataset(real_train, fake_train, tokenizer, padding_max_length)
        train_loader = DataLoader(train_dataset, batch_size, sampler=train_sampler, num_workers=0)

        valid_dataset = TextDataset(real_valid, fake_valid, tokenizer, padding_max_length)
        valid_loader = DataLoader(valid_dataset, 1, shuffle= True, num_workers=0)
        return train_loader, valid_loader
    else:
        return real_test, fake_test

        