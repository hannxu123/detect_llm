import numpy as np
import json
from typing import List
from transformers import PreTrainedTokenizer
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix



class Corpus:
    def __init__(self, dataset, train_ratio=0.8, valid_ratio=0.1):
        self.length_train = int(len(dataset) * train_ratio)
        self.length_valid = int(len(dataset) * valid_ratio)
        self.train = dataset[:self.length_train]
        self.valid = dataset[self.length_train:self.length_train+self.length_valid]
        self.test = dataset[self.length_train+self.length_valid:]


class EncodedDataset(Dataset):
    def __init__(self, real_texts: List[str], fake_texts: List[str], tokenizer: PreTrainedTokenizer, max_sequence_length: int = 512):
        self.real_texts = real_texts
        self.fake_texts = fake_texts
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length

    def __len__(self):
        return len(self.real_texts) + len(self.fake_texts)

    def __getitem__(self, index):
        if index < len(self.real_texts):
            text = self.real_texts[index]
            label = 1
        else:
            text = self.fake_texts[index - len(self.real_texts)]
            label = 0

        tokenized_test = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_sequence_length, return_tensors='pt')

        return tokenized_test['input_ids'][0], tokenized_test['attention_mask'][0], label


# load texts from a JSON file
def load_texts(data_file):
    with open(data_file) as f:
        data = [json.loads(line)['text'] for line in f]
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


def cut_text(text, long_text_length=32):
    text = [x.strip() for x in text]
    text = [strip_newlines(x) for x in text]
    text = [process_spaces(x) for x in text]
    long_text = [x for x in text if (len(x.split()) > long_text_length)]
    return long_text


def accuracy_sum(logits, labels):
    if list(logits.shape) == list(labels.shape) + [2]:
        # 2-d outputs
        classification = (logits[..., 0] < logits[..., 1]).long().flatten()
    else:
        classification = (logits > 0).long().flatten()
    assert classification.shape == labels.shape
    return (classification == labels).float().sum().item()


def load_datasets(data, train=False, tokenizer=None, batch_size=1, max_sequence_length=512):
    original_dataset = data['original']
    sampled_dataset = data['sampled']

    original_corpus = Corpus(original_dataset)
    sampled_corpus = Corpus(sampled_dataset)

    original_train, original_valid, original_test = original_corpus.train, original_corpus.valid, original_corpus.test
    sampled_train, sampled_valid, sampled_test = sampled_corpus.train, sampled_corpus.valid, sampled_corpus.test

    if train:
        train_dataset = EncodedDataset(original_train, sampled_train, tokenizer, max_sequence_length)
        weight = torch.cat([len(sampled_train) / len(original_train) * torch.ones(len(original_train)), torch.ones(len(sampled_train))])
        train_sampler = WeightedRandomSampler(weight, len(original_train) + len(sampled_train), replacement=False)
        train_loader = DataLoader(train_dataset, batch_size, sampler=train_sampler, num_workers=0)

        validation_dataset = EncodedDataset(original_valid, sampled_valid, tokenizer)
        validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=True, num_workers=0)

        return train_loader, validation_loader
    else:
        return original_test, sampled_test


def calc_metrics(orig_preds, sampled_preds):
    orig_pred_labels = [round(pred) for pred in orig_preds]
    sampled_pred_labels = [round(pred) for pred in sampled_preds]

    all_orig_labels = [0] * len(orig_preds) + [1] * len(sampled_preds)
    all_pred_labels = orig_pred_labels + sampled_pred_labels

    acc = accuracy_score(all_orig_labels, all_pred_labels)
    precision = precision_score(all_orig_labels, all_pred_labels)
    recall = recall_score(all_orig_labels, all_pred_labels)
    f1 = f1_score(all_orig_labels, all_pred_labels)
    auc = roc_auc_score(all_orig_labels, orig_preds + sampled_preds)

    tn, fp, fn, tp = confusion_matrix(all_orig_labels, all_pred_labels).ravel()
    tpr = tp/(tp+fn)
    fpr = fp/(fp+tn)

    return acc, precision, recall, f1, auc, tpr, 1-fpr

