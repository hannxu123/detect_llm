
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
    random.shuffle(long_data)
    return long_data

def Corpus_imdb(prompt = 'p1', train_name = 'imdb'):

    fake_data = []
    file_path = 'data_save/' + train_name + '_' + prompt + '_results.jsonl'
    jsonl_data = load_jsonl(file_path)

    for message in jsonl_data:
        try:
            fake_data.append(message[1]['choices'][0]['message']['content'])
        except:
            # print('Identify 1 record ChatGPT Failure')
            fake_data = fake_data
    fake_data = cut_data(fake_data)

    ## processing human data
    if train_name == 'imdb':
        real_dataset = load_dataset("imdb")['train']
        real_data = []
        for i in range(10000):   ## select 10000 imdb original review as human written
            real_data.append(real_dataset[i]['text'])
        real_data = cut_data(real_data)

    elif train_name == 'amazon':
        # real_data = []
        # topics= ['Apparel_v1_00', 'Electronics_v1_00', 'Software_v1_00', 'Sports_v1_00', 'Toys_v1_00']
        # for t in topics:
        #     dataset = load_dataset('amazon_us_reviews', t)['train']
        #     n = 0
        #     for dat in dataset:
        #         if len(dat['review_body'].split()) > 32:
        #             real_data.append(dat['review_body'])
        #             n = n + 1
        #             if n > 2000:
        #                 print('Done collecting reviews from ' + t)
        #                 break
        # with open('data_save/real_amazon', "wb") as fp1:  # Pickling
        #     pickle.dump(real_data, fp1)
        with open('data_save/real_amazon', "rb") as fp:  # Unpickling
            real_data = pickle.load(fp)
    elif train_name == 'eli5':
        # real_data = []
        # n = 0
        # data = load_dataset("eli5")['train_eli5']
        # for dat in data:
        #     n = n + 1
        #     dat_list = dat['answers']['text']
        #     for j in dat_list:
        #         if len(j.split()) > 32:
        #             real_data.append(j)
        #             break
        #     if n > 10000:
        #         break
        # with open('data_save/real_eli5', "wb") as fp1:  # Pickling
        #     pickle.dump(real_data, fp1)
        with open('data_save/real_eli5', "rb") as fp:  # Unpickling
            real_data = pickle.load(fp)
    else:
        raise ValueError

    return real_data, fake_data



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

def loader(batch_size, prompt = 'p1', train_name = 'imdb'):
    real_data, fake_data = Corpus_imdb(prompt, train_name = train_name)
    real_train = real_data[0:len(real_data) - 400]
    real_valid = real_data[len(real_data) - 400:len(real_data) - 250]
    real_test = real_data[len(real_data) - 250:]

    fake_train = fake_data[0:len(fake_data) - 400]
    fake_valid = fake_data[len(fake_data) - 400:len(fake_data) - 250]
    fake_test = fake_data[len(fake_data) - 250:]

    weight = torch.cat(
        [len(fake_train) / len(real_train) * torch.ones(len(real_train)), torch.ones(len(fake_train))])
    train_sampler = WeightedRandomSampler(weight, len(real_train) + len(fake_train), replacement=False)

    train_dataset = TextDataset(real_train, fake_train)
    train_loader = DataLoader(train_dataset, batch_size, sampler=train_sampler, num_workers=0)

    test_dataset = TextDataset(real_test, fake_test)
    test_loader = DataLoader(test_dataset, 1, shuffle=True, num_workers=0)

    valid_dataset = TextDataset(real_valid, fake_valid)
    valid_loader = DataLoader(valid_dataset, 1, shuffle=True, num_workers=0)

    return train_loader, valid_loader, test_loader


def loader_all(batch_size, prompt='all2',train_name = 'imdb'):
    fake_train = []
    fake_valid = []
    fake_test = []
    for j in ['p1', 'p2', 'p3']:
        real_data, fake_temp = Corpus_imdb(j, train_name = train_name)
        if prompt == 'all1':
            fake_train.extend(fake_temp[0:900])
        else:
            fake_train.extend(fake_temp[0:len(fake_temp) - 400])
        fake_valid.extend(fake_temp[len(fake_temp) - 400:len(fake_temp) - 250])
        fake_test.extend(fake_temp[len(fake_temp) - 250:])

    real_train = real_data[0:len(real_data) - 400]
    real_valid = real_data[len(real_data) - 400:len(real_data) - 250]
    real_test = real_data[len(real_data) - 250:]

    weight = torch.cat(
        [len(fake_train) / len(real_train) * torch.ones(len(real_train)), torch.ones(len(fake_train))])
    train_sampler = WeightedRandomSampler(weight, len(real_train) + len(fake_train), replacement=False)

    train_dataset = TextDataset(real_train, fake_train)
    train_loader = DataLoader(train_dataset, batch_size, sampler=train_sampler, num_workers=0)

    valid_dataset = TextDataset(real_valid, fake_valid)
    valid_loader = DataLoader(valid_dataset, 1, shuffle=True, num_workers=0)

    test_dataset = TextDataset(real_test, fake_test[0:250])
    test_loader1 = DataLoader(test_dataset, 1, shuffle=True, num_workers=0)

    test_dataset = TextDataset(real_test, fake_test[250:500])
    test_loader2 = DataLoader(test_dataset, 1, shuffle=True, num_workers=0)

    test_dataset = TextDataset(real_test, fake_test[500:])
    test_loader3 = DataLoader(test_dataset, 1, shuffle=True, num_workers=0)

    return train_loader, valid_loader, test_loader1, test_loader2, test_loader3


