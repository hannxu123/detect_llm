# https://github.com/Hello-SimpleAI/chatgpt-comparison-detection/blob/main/detect/ml_train.py
import math
import numpy as np
import os
import json
import pickle
import random
from functools import partial
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
import torch
import transformers
from datasets import load_dataset
import params
import importlib
importlib.reload(params)
from utils import *


def data_to_xy(dataset):
    return np.asarray(dataset['x'][0]), np.asarray(dataset['labels'][0])


def load_dataset_for_gltr_classifier(data):
    original_dataset = data['original']
    sampled_dataset = data['sampled']

    original_corpus = Corpus(original_dataset)
    sampled_corpus = Corpus(sampled_dataset)

    original_train, original_valid, original_test = original_corpus.train, original_corpus.valid, original_corpus.test
    sampled_train, sampled_valid, sampled_test = sampled_corpus.train, sampled_corpus.valid, sampled_corpus.test

    return original_train, sampled_train, original_test, sampled_test


def construct_dataset(original_train, sampled_train, cache_dir, shuffle=True):
    final_text = []
    final_label = []
    final_text.extend(original_train)
    final_text.extend(sampled_train)
    final_label.extend([0] * len(original_train))
    final_label.extend([1] * len(sampled_train))

    if shuffle:
        # random shuffle
        temp = list(zip(final_text, final_label))
        random.shuffle(temp)
        res1, res2 = zip(*temp)
        # res1 and res2 come out as tuples, and so must be converted to lists.
        new_texts, new_labels = list(res1), list(res2)

        data_dict = {'texts': new_texts, 
            'labels': new_labels
        } 
    else:
        data_dict = {'texts': final_text, 
            'labels': final_label
        } 

    dataset = Dataset.from_dict(data_dict)

    return dataset


def train_lr_classifier(x, y, save_path):
    # borrowed from https://github.com/jmpu/DeepfakeTextDetection
    # define models and parameters
    model = LogisticRegression()
    solvers = ['newton-cg', 'lbfgs', 'liblinear']
    penalty = ['l2']
    c_values = [100, 10, 1.0, 0.1, 0.01]
    # define grid search
    grid = dict(solver=solvers, penalty=penalty, C=c_values)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy', error_score=0)
    grid_result = grid_search.fit(x, y)
    model_train = grid_result.best_estimator_

    return model_train


def predict_lr_classifier(model, x, y):
    # borrowed from https://github.com/jmpu/DeepfakeTextDetection
    #y_pred = model.predict(x)
    y_prob = model.predict_proba(x)
    y_prod_pos = y_prob[:,1]

    orig_preds = []
    sampled_preds = []
    for i in range(len(y)):
        if y[i] == 0:
            orig_preds.append(y_prod_pos[i])
        else:
            sampled_preds.append(y_prod_pos[i])

    return orig_preds, sampled_preds


def gltr_batched(batch, model, tokenizer, device, padding_max_length, batch_size):
    """
    Batched rank buckets computation of GLTR.
    """
    encoded = tokenizer.batch_encode_plus(batch['texts'][0], return_tensors='pt', padding='max_length', max_length=min(padding_max_length, tokenizer.model_max_length-2), truncation=True).data
    
    final_stat = []
    for cur_batch in range(math.ceil(len(encoded['input_ids']) / batch_size)): 
        try:
            batch_input_ids = encoded['input_ids'][cur_batch * batch_size:(cur_batch + 1) * batch_size]
            batch_mask = encoded['attention_mask'][cur_batch * batch_size:(cur_batch + 1) * batch_size]
        except:
            batch_input_ids = encoded['input_ids'][cur_batch * batch_size:len(encoded['input_ids'])]
            batch_mask = encoded['attention_mask'][cur_batch * batch_size:len(encoded['input_ids'])]

        input_ids, mask = batch_input_ids.to(device), batch_mask.to(device)
        bos = input_ids.new_full((mask.size(0), 1), tokenizer.bos_token_id)
        input_dict = dict(
            input_ids=torch.cat([bos, input_ids], dim=1),
            attention_mask=torch.cat([mask.new_ones((mask.size(0), 1)), mask], dim=1)
        )
        output = model(**input_dict)

        all_logits = output.logits[:, :-1]  # n-1 predict n
        all_probs = torch.softmax(all_logits, dim=-1)
        sorted_ids = torch.argsort(all_probs, dim=-1, descending=True)  # stable=True
        expanded_tokens = input_ids.unsqueeze(-1).expand_as(sorted_ids)
        indices = torch.where(sorted_ids == expanded_tokens)
        rank = indices[2]
        counter = [
            rank < 10,
            (rank >= 10) & (rank < 100),
            (rank >= 100) & (rank < 1000),
            rank >= 1000
        ]
        counter = [c.long().reshape(input_ids.size()).mul_(mask).sum(-1, keepdim=True) for c in counter]
        final_stat.extend(torch.cat(counter, dim=-1).tolist())

    batch['x'] = final_stat

    return batch


def predict_data(dataset, data_path, base_model, base_tokenizer, device, batch_size, padding_max_length):
    base_tokenizer.pad_token = base_tokenizer.eos_token
    base_model.eval()
    kwargs = dict(model=base_model, tokenizer=base_tokenizer, device=device, padding_max_length=padding_max_length, batch_size=batch_size)

    processor = partial(gltr_batched, **kwargs)

    with torch.no_grad():
        dataset = dataset.map(processor, batched=True, batch_size=batch_size)
    dataset.to_json(data_path, orient='records', lines=True, force_ascii=False)

    return dataset


class GLTR_classifier:
    def __init__(self, model_name, dataset_name, cache_dir, device):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.cache_dir = cache_dir
        self.device = device
        self.base_model = transformers.GPT2LMHeadModel.from_pretrained('gpt2', cache_dir=self.cache_dir) 
        self.base_tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2', cache_dir=self.cache_dir)
        self.base_model = self.base_model.to(self.device)


    def train_and_test_func(self, data):
        save_path = self.cache_dir + '/' + self.model_name 
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        original_train, sampled_train, _, _ = load_dataset_for_gltr_classifier(data)
        init_dataset = construct_dataset(original_train, sampled_train, self.cache_dir, True)
        used_dataset = predict_data(init_dataset, data_path, self.base_model, self.base_tokenizer, self.device, params.OTHERS_PARAMS['test_batch_size'], params.OTHERS_PARAMS['padding_max_length'])

        x, y = data_to_xy(used_dataset)
        # train GLTR classifier
        self.classification_model = train_lr_classifier(x, y)

        # test GLTR classifier
        _, _, original_test, sampled_test = load_dataset_for_gltr_classifier(data)
        init_dataset = construct_dataset(original_test, sampled_test, self.cache_dir, False)
        used_dataset = predict_data(init_dataset, data_path, self.base_model, self.base_tokenizer, self.device, params.OTHERS_PARAMS['test_batch_size'], params.OTHERS_PARAMS['padding_max_length'])

        x, y = data_to_xy(used_dataset)

        orig_preds, sampled_preds = predict_lr_classifier(self.classification_model, x, y)

        acc, precision, recall, f1, auc, tpr, fpr = calc_metrics(orig_preds, sampled_preds)
        print(f'{self.model_name} test Acc: {acc}, test Precision: {precision}, test Recall: {recall}, test F1: {f1}, test AUC: {auc}, test TPR: {tpr}, test FPR: {fpr}', flush=True)




