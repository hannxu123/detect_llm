import math
import numpy as np
import os
from typing import List
import json
from itertools import count
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Function, Variable
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torch.optim import AdamW
from transformers import PreTrainedTokenizer
from utils import *


class MPG:
    def __init__(self, pretrained_model=True, n_samples=500):
        self.pretrained_model = pretrained_model
        self.n_samples = n_samples

    def main_func(self, data, model_collector, results_path, batch_size, pretrained_model_name):
        output = run_detect_experiment(data, pretrained_model_name, model_collector, batch_size, self.n_samples)
        with open(os.path.join(results_path, 'results.json'), 'w') as f:
            json.dump(output, f)

    def train_func(self, data, model_collector, cache_dir='./cache_dir', device='cuda', results_path='', learning_rate=5e-5, weight_decay=0,
                   batch_size=8, max_epochs=1, max_sequence_length=512, epoch_size=None, prior=0.2, pu_type='dual_softmax_dyn_dtrun', lamb=0.4, len_thres=55):
        
        original_text = data['original']
        sampled_text = data['sampled']

        train_loader, validation_loader = load_datasets(original_text, sampled_text, cache_dir, model_collector.classification_tokenizer, batch_size, max_sequence_length, epoch_size)

        optimizer = AdamW(model_collector.classification_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        epoch_loop = count(1) if max_epochs is None else range(1, max_epochs + 1)

        save_path = cache_dir + '/mpg/'

        if not os.path.exists(save_path):
            os.mkdir(save_path)

        best_validation_accuracy = 0

        for epoch in epoch_loop:
            train_loss, train_acc = train(model_collector.classification_model, optimizer, device, train_loader, prior, pu_type, lamb, len_thres)
            print(f'Training epoch {epoch} --- Training loss: {train_loss}, Training accuracy: {train_acc}', flush=True)
            val_loss, val_acc = validate(model_collector.classification_model, device, validation_loader)
            print(f'Training epoch {epoch} --- Validation loss: {val_loss}, Validation accuracy: {val_acc}', flush=True)
            
            if val_acc > best_validation_accuracy:
                best_validation_accuracy = val_acc

                model_to_save = model_collector.classification_model.module if hasattr(model_collector.classification_model, 'module') else model_collector.classification_model
                torch.save(dict(
                        epoch=epoch,
                        model_state_dict=model_to_save.state_dict(),
                        optimizer_state_dict=optimizer.state_dict()
                    ),
                    os.path.join(save_path, 'best_model.pt')
                )

            output = run_detect_experiment(data, pretrained_model_name, model_collector, batch_size, self.n_samples)
            with open(os.path.join(results_path, 'results.json'), 'w') as f:
                json.dump(output, f)



def train(model, optimizer, device, loader, prior, pu_type, lamb, len_thres):
    model.train()

    module = None
    if lamb > 0: # prepare pu module
        module = pu_loss_auto(prior, pu_type, device=device)

    train_accuracy = 0
    train_epoch_size = 0
    train_loss = 0

    for texts, masks, labels in loader:
        texts, masks, labels = texts.to(device), masks.to(device), labels.to(device)
        batch_size = texts.shape[0]
        
        optimizer.zero_grad()
        results = model(texts, attention_mask=masks, labels=labels) 
        loss, logits = results['loss'], results['logits'] 

        if lamb > 0: # PU loss activated, self added
            # self added: process short sentence labels: set as -1
            # filter out positive and unlabeled
            pad_id = model.module.config.pad_token_id if hasattr(model, 'module') else model.config.pad_token_id
            sentence_length = (texts!=pad_id).sum(dim=-1) # calc length
            pseudo_labels = (~labels.bool()).float()
            U_mask = (sentence_length < len_thres) & (labels.bool()) # select short, chatgpt sentences as unlabeled
            P_long_mask = (sentence_length < len_thres) & (~labels.bool()) # long human sentences
            pseudo_labels[U_mask] = -1
            pseudo_labels[P_long_mask] = 0 # disregard long human corpus
            # calc pu loss
            scores = module.logits_to_scores(logits)
            puloss = module(scores, pseudo_labels, sentence_length)
            loss += lamb * puloss
            # save sentence lengths to folder
            #args.sentence_lengths.append(sentence_length.cpu())
        loss.backward()
        optimizer.step()

        batch_accuracy = accuracy_sum(torch.softmax(logits, dim=-1), labels)
        train_accuracy += batch_accuracy
        train_epoch_size += batch_size
        train_loss += loss.item() * batch_size

    return train_loss / train_epoch_size, train_accuracy / train_epoch_size


def validate(model, device, loader):
    model.eval()

    validation_accuracy = 0
    validation_epoch_size = 0
    validation_loss = 0

    with torch.no_grad():
        for texts, masks, labels in loader:
            texts, masks, labels = texts.to(device), masks.to(device), labels.to(device)
            batch_size = texts.shape[0]

            results = model(texts, attention_mask=masks, labels=labels) 
            loss, logits = results['loss'], results['logits']

        batch_accuracy = accuracy_sum(torch.softmax(logits, dim=-1), labels)
        validation_accuracy += batch_accuracy
        validation_epoch_size += batch_size
        validation_loss += loss.item() * batch_size

    return validation_loss / validation_epoch_size, validation_accuracy / validation_epoch_size



def accuracy_sum(logits, labels):
    if list(logits.shape) == list(labels.shape) + [2]:
        # 2-d outputs
        classification = (logits[..., 0] < logits[..., 1]).long().flatten()
    else:
        classification = (logits > 0).long().flatten()
    assert classification.shape == labels.shape
    return (classification == labels).float().sum().item()


def load_datasets(real_dataset, fake_dataset, cache_dir, tokenizer, batch_size, max_sequence_length, epoch_size=None):
    #real_dataset, fake_dataset, _ = get_hc3_dataset('hc3-english', cache_dir)

    real_corpus = Corpus(real_dataset, cache_dir=cache_dir)
    fake_corpus = Corpus(fake_dataset, cache_dir=cache_dir)

    real_train, real_valid = real_corpus.train, real_corpus.valid
    fake_train, fake_valid = fake_corpus.train, fake_corpus.valid

    train_dataset = EncodedDataset(real_train, fake_train, tokenizer, max_sequence_length, epoch_size)
    train_loader = DataLoader(train_dataset, batch_size, sampler=RandomSampler(train_dataset), num_workers=0)

    validation_dataset = EncodedDataset(real_valid, fake_valid, tokenizer)
    validation_loader = DataLoader(validation_dataset, batch_size=1, sampler=RandomSampler(validation_dataset))

    return train_loader, validation_loader


def run_detect_experiment(data, pretrained_model_name, model_collector, batch_size, n_samples=500):
    results = []
    original_text = data["original"][:n_samples]
    sampled_text = data["sampled"][:n_samples]

    if pretrained_model_name == 't5-sentinel':
        positive_token_id = model_collector.classification_tokenizer('positive', return_tensors='pt')['input_ids'][0][0].item()  # = 1465, refers to ChatGPT-generated text 
        negative_token_id = model_collector.classification_tokenizer('negative', return_tensors='pt')['input_ids'][0][0].item()  # = 2841, refers to human-written text 

    with torch.no_grad():
        # get predictions for original text
        real_preds = []
        for batch in range(math.ceil(len(original_text) / batch_size)): 
            try:
                batch_real = original_text[batch * batch_size:(batch + 1) * batch_size]
            except:
                batch_real = original_text[batch * batch_size:len(original_text)]

            tokenized_batch_real = model_collector.classification_tokenizer(batch_real, padding=True, truncation=True, max_length=512, return_tensors="pt").to(model_collector.classification_model.device)

            if pretrained_model_name == 't5-sentinel':
                decoder_input_ids = torch.tensor([model_collector.classification_tokenizer.pad_token_id] * len(tokenized_batch_real['input_ids'])).unsqueeze(-1).to(model_collector.classification_model.device)
                logits = model_collector.classification_model(input_ids=tokenized_batch_real['input_ids'], decoder_input_ids=decoder_input_ids)
                logits = logits[0].squeeze(1)  # output dimension [len(tokenized_batch_real), 1, 32128]
                selected_logits = logits[:, [positive_token_id, negative_token_id]] 
                pred_probs = torch.softmax(selected_logits, dim=-1)
                real_preds.extend(pred_probs[:,0].cpu().numpy().tolist())
            else:
                pred_probs = model_collector.classification_model(input_ids=tokenized_batch_real['input_ids'], attention_mask=tokenized_batch_real['attention_mask'])
                pred_probs = torch.softmax(pred_probs, dim=-1)
                real_preds.extend(pred_probs[:,0].detach().cpu().numpy().tolist())
        
        # get predictions for sampled text
        fake_preds = []
        for batch in range(math.ceil(len(sampled_text) / batch_size)):
            try:
                batch_fake = sampled_text[batch * batch_size:(batch + 1) * batch_size]
            except:
                batch_fake = sampled_text[batch * batch_size:len(sampled_text)]

            tokenized_batch_fake = model_collector.classification_tokenizer(batch_fake, padding=True, truncation=True, max_length=512, return_tensors="pt").to(model_collector.classification_model.device)

            if pretrained_model_name == 't5-sentinel':
                decoder_input_ids = torch.tensor([model_collector.classification_tokenizer.pad_token_id] * len(tokenized_batch_fake['input_ids'])).unsqueeze(-1).to(model_collector.classification_model.device)
                logits = model_collector.classification_model(input_ids=tokenized_batch_fake['input_ids'], decoder_input_ids=decoder_input_ids)
                logits = logits[0].squeeze(1)  # output dimension [len(tokenized_batch_fake), 1, 32128]
                selected_logits = logits[:, [positive_token_id, negative_token_id]] 
                pred_probs = torch.softmax(selected_logits, dim=-1)
                fake_preds.extend(pred_probs[:,0].cpu().numpy().tolist())
            else:
                pred_probs = model_collector.classification_model(input_ids=tokenized_batch_fake['input_ids'], attention_mask=tokenized_batch_fake['attention_mask'])
                pred_probs = torch.softmax(pred_probs, dim=-1)
                fake_preds.extend(pred_probs[:,0].detach().cpu().numpy().tolist())

    predictions = {
        'real': real_preds,
        'samples': fake_preds,
    }

    fpr, tpr, roc_auc = get_roc_metrics(real_preds, fake_preds)
    p, r, pr_auc = get_precision_recall_metrics(real_preds, fake_preds)
    print(f"ROC AUC: {roc_auc}, PR AUC: {pr_auc}")

    return {
        'predictions': predictions,
        'info': {
            'n_samples': n_samples,
        },
        'metrics': {
            'roc_auc': roc_auc,
            'fpr': fpr,
            'tpr': tpr,
        },
        'pr_metrics': {
            'pr_auc': pr_auc,
            'precision': p,
            'recall': r,
        },
        'loss': 1 - pr_auc,
    }


def expectation_matrix(length, pi, device='cpu'):
    if length < 3:
        return torch.tensor(pi).float().to(device)
    state = torch.zeros((1, length+1)).float().to(device)
    state[0, 0] += 1.
    trans = torch.zeros((length+1,length+1)).float().to(device) # state transition matrix
    trans[1:, :-1] += torch.eye(length).to(device)*pi
    trans[:-1, 1:] += torch.eye(length).to(device)*(1-pi)
    trans[0,0] += pi
    trans[length, length] += (1-pi)

    total_trans = torch.zeros_like(trans) + torch.eye(length+1).to(device) # id mat
    for _ in range(length):
        total_trans @= trans
    distribution = (state @ total_trans).squeeze(0)
    expectation = 1. - ((distribution * torch.arange(0, length+1).to(device)).sum()/length)
    return expectation.to(device)


class PULossauto:
    def __init__(self):
        self.prior = 0
        self.label = 0

    # @staticmethod
    def apply(self, input, label, prior):
        self.input = input
        self.label = label
        if type(prior)==float: # self added
            prior = torch.tensor(prior)
        self.prior = prior.to(input.device).float()
        self.positive = 1
        self.unlabeled = -1
        self.loss_func = lambda x: F.sigmoid(-x) # this x is merely a real number
        self.beta = 0
        self.gamma = 1
        
        self.positive_x = (self.label==self.positive).float()
        self.unlabeled_x = (self.label==self.unlabeled).float()
        self.positive_num = torch.max(torch.sum(self.positive_x), torch.tensor(1).to(input.device).float())
        self.unlabeled_num = torch.max(torch.sum(self.unlabeled_x), torch.tensor(1).to(input.device).float())
        self.positive_y = self.loss_func(self.input)
        self.unlabeled_y = self.loss_func(-self.input) # all regarded as negative
        self.positive_loss = torch.sum(self.prior * self.positive_x / self.positive_num * self.positive_y.squeeze())
        self.negative_loss = torch.sum((self.unlabeled_x / self.unlabeled_num - self.prior * self.positive_x / self.positive_num) * self.unlabeled_y.squeeze())
        objective = self.positive_loss + self.negative_loss
        
        if self.negative_loss.data < -self.beta:
            objective = self.positive_loss - self.beta
            self.x_out = -self.gamma * self.negative_loss
        else:
            self.x_out = objective
        return objective


class pu_loss_auto():
    def __init__(self, prior, pu_type='', max_length=512, device='cpu'):
        self.prior = prior
        # self.label = label
        self.pu_type = pu_type
        self.device = device
        if pu_type in ['dual_softmax_dyn_dtrun']:
            self.loss_mod = PULossauto()
        else:
            raise NotImplementedError(f'PU type {pu_type} not implemented...')
        # for random walk:
        if pu_type in ['dual_softmax_dyn_dtrun']:
            expectations = list()
            for i in range(0, max_length+1):
                expectations.append(expectation_matrix(i, self.prior, device))
            self.prior = torch.stack(expectations)
            print('All dynamic priors calculated...')


    def __call__(self, input, label, sentence_length):
        prior = self.prior
        if 'dyn' in self.pu_type:
            prior = self.prior[sentence_length]
        return self.loss_mod.apply(input, label, prior)
    
    def logits_to_scores(self, logits):
        if self.pu_type in ['dual_softmax_dyn_dtrun']:
            return F.softmax(logits, dim=-1)[..., 0] # take human as positive
        else:
            raise NotImplementedError(f'PU type {self.pu_type} not implemented')


class Corpus:
    def __init__(self, dataset, cache_dir, skip_train=False):
        self.length_train = int(len(dataset) * 0.8)
        self.length_valid = int(len(dataset) * 0.1)
        self.train = dataset[:self.length_train]
        self.valid = dataset[self.length_train:self.length_train+self.length_valid]
        self.test = dataset[self.length_train+self.length_valid:]


class EncodedDataset(Dataset):
    def __init__(self, real_texts: List[str], fake_texts: List[str], tokenizer: PreTrainedTokenizer,
                 max_sequence_length: int = 512, epoch_size: int = None):
        self.real_texts = real_texts
        self.fake_texts = fake_texts
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
        self.epoch_size = epoch_size

    def __len__(self):
        return self.epoch_size or len(self.real_texts) + len(self.fake_texts)

    def __getitem__(self, index):
        if self.epoch_size is not None:
            label = self.random.randint(2)
            texts = [self.fake_texts, self.real_texts][label]
            text = texts[self.random.randint(len(texts))]
        else:
            if index < len(self.real_texts):
                text = self.real_texts[index]
                label = 1
            else:
                text = self.fake_texts[index - len(self.real_texts)]
                label = 0

        tokens = self.tokenizer.encode(text, truncation=True, max_length=self.max_sequence_length)

        padding = [self.tokenizer.pad_token_id] * (self.max_sequence_length - len(tokens))
        tokens = torch.tensor(tokens + padding)
        mask = torch.ones(tokens.shape[0])
        mask[-len(padding):] = 0
        return tokens, mask, label

