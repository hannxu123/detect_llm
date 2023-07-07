import math
import numpy as np
import os
from typing import List
import json
from itertools import count
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torch.optim import AdamW
from transformers import PreTrainedTokenizer
from utils import *


class HC3_classifiers:
    def __init__(self, pretrained_model=True, n_samples=500):
        self.pretrained_model = pretrained_model
        self.n_samples = n_samples

    def main_func(self, data, model_collector, results_path, batch_size, pretrained_model_name):
        output = run_detect_experiment(data, pretrained_model_name, model_collector, batch_size, self.n_samples)
        with open(os.path.join(results_path, 'results.json'), 'w') as f:
            json.dump(output, f)

    def train_func(self, model_collector, cache_dir='./cache_dir', device='cuda', data='xsum', results_path='', pretrained_model_name='roberta-sentinel', learning_rate=5e-5, weight_decay=1e-3,
                   batch_size=32, max_epochs=1, max_sequence_length=512, epoch_size=None, loss_func=torch.nn.CrossEntropyLoss()):
        
        train_loader, validation_loader = load_datasets(cache_dir, model_collector.classification_tokenizer, batch_size, max_sequence_length, epoch_size)

        optimizer = AdamW(model_collector.classification_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        epoch_loop = count(1) if max_epochs is None else range(1, max_epochs + 1)

        save_path = cache_dir + '/hc3_classifier/'

        if not os.path.exists(save_path):
            os.mkdir(save_path)

        best_validation_accuracy = 0

        for epoch in epoch_loop:
            train_loss, train_acc = train_roberta(model_collector.classification_model, optimizer, device, train_loader, loss_func, pretrained_model_name)
            print(f'Training epoch {epoch} --- Training loss: {train_loss}, Training accuracy: {train_acc}', flush=True)
            val_loss, val_acc = validate_roberta(model_collector.classification_model, device, validation_loader, loss_func, pretrained_model_name)
            print(f'Training epoch {epoch} --- Validation loss: {val_loss}, Validation accuracy: {val_acc}', flush=True)
            
            if val_acc > best_validation_accuracy:
                best_validation_accuracy = val_acc

                model_to_save = model_collector.classification_model.module if hasattr(model_collector.classification_model, 'module') else model_collector.classification_model
                torch.save(dict(
                        epoch=epoch,
                        model_state_dict=model_to_save.state_dict(),
                        optimizer_state_dict=optimizer.state_dict()
                    ),
                    os.path.join(save_path, pretrained_model_name +'_best_model.pt')
                )

            output = run_detect_experiment(data, pretrained_model_name, model_collector, batch_size, self.n_samples)
            with open(os.path.join(results_path, 'results.json'), 'w') as f:
                json.dump(output, f)

            scheduler.step()

def train_roberta(model, optimizer, device, loader, loss_func, model_name):
    model.train()

    train_accuracy = 0
    train_epoch_size = 0
    train_loss = 0

    for texts, masks, labels in loader:
        texts, masks, labels = texts.to(device), masks.to(device), labels.to(device)
        batch_size = texts.shape[0]

        optimizer.zero_grad()
        if model_name == 'roberta-single':
            logits = model(texts, attention_mask=masks)
        else:
            logits = model(texts, attention_mask=masks)
        loss = loss_func(logits, labels)
        loss.backward()
        optimizer.step()

        batch_accuracy = accuracy_sum(torch.softmax(logits, dim=-1), labels)
        train_accuracy += batch_accuracy
        train_epoch_size += batch_size
        train_loss += loss.item() * batch_size

    return train_loss / train_epoch_size, train_accuracy / train_epoch_size


def validate_roberta(model, device, loader, loss_func, model_name):
    model.eval()

    validation_accuracy = 0
    validation_epoch_size = 0
    validation_loss = 0

    with torch.no_grad():
        for texts, masks, labels in loader:
            texts, masks, labels = texts.to(device), masks.to(device), labels.to(device)
            batch_size = texts.shape[0]

            if model_name == 'roberta-single':
                logits = model(texts, attention_mask=masks)
            else:
                logits = model(texts, attention_mask=masks)
            loss = loss_func(logits, labels)

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


def load_datasets(cache_dir, tokenizer, batch_size, max_sequence_length, epoch_size=None):
    real_dataset, fake_dataset, question_dataset = get_hc3_dataset(cache_dir)

    real_corpus = Corpus(real_dataset, cache_dir=cache_dir)
    fake_corpus = Corpus(fake_dataset, cache_dir=cache_dir)
    question_corpus = Corpus(question_dataset, cache_dir=cache_dir)

    real_train, real_valid = real_corpus.train, real_corpus.valid
    fake_train, fake_valid = fake_corpus.train, fake_corpus.valid
    question_train, question_valid = fake_corpus.train, fake_corpus.valid

    train_dataset = EncodedDataset(real_train, fake_train, question_train, tokenizer, max_sequence_length, epoch_size)
    train_loader = DataLoader(train_dataset, batch_size, sampler=RandomSampler(train_dataset), num_workers=0)

    validation_dataset = EncodedDataset(real_valid, fake_valid, question_valid, tokenizer)
    validation_loader = DataLoader(validation_dataset, batch_size=1, sampler=RandomSampler(validation_dataset))

    return train_loader, validation_loader


def run_detect_experiment(data, pretrained_model_name, model_collector, batch_size, n_samples=500):
    results = []
    original_text = data["original"][:n_samples]
    sampled_text = data["sampled"][:n_samples]

    with torch.no_grad():
        # get predictions for original text
        real_preds = []
        for batch in range(math.ceil(len(original_text) / batch_size)): 
            try:
                batch_real = original_text[batch * batch_size:(batch + 1) * batch_size]
            except:
                batch_real = original_text[batch * batch_size:len(original_text)]

            tokenized_batch_real = model_collector.classification_tokenizer(batch_real, padding=True, truncation=True, max_length=512, return_tensors="pt").to(model_collector.classification_model.device)

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


class Corpus:
    def __init__(self, dataset, cache_dir, train_ratio=0.8, valid_ratio=0.1, skip_train=False):
        self.length_train = int(len(dataset) * train_ratio)
        self.length_valid = int(len(dataset) * valid_ratio)
        self.train = dataset[:self.length_train]
        self.valid = dataset[self.length_train:self.length_train+self.length_valid]
        self.test = dataset[self.length_train+self.length_valid:]  


class EncodedDataset(Dataset):
    def __init__(self, real_original_texts: List[str], real_generate_texts: List[str], fake_original_texts: List[str], fake_generate_texts: List[str], tokenizer: PreTrainedTokenizer,
                 max_sequence_length: int = 512, epoch_size: int = None):
        self.real_texts = real_original_texts
        self.real_texts_gene = real_generate_texts
        self.fake_texts = fake_original_texts
        self.fake_texts_gene = fake_generate_texts
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
                text = (self.real_texts[index], self.real_texts_gene[index])
                label = 1
            else:
                text = (self.fake_texts[index - len(self.real_texts)], self.fake_texts_gene[index - len(self.real_texts)])
                label = 0

        tokens = self.tokenizer.encode(text, truncation=True, max_length=self.max_sequence_length)

        padding = [self.tokenizer.pad_token_id] * (self.max_sequence_length - len(tokens))
        tokens = torch.tensor(tokens + padding)
        mask = torch.ones(tokens.shape[0])
        mask[-len(padding):] = 0
        return tokens, mask, label

