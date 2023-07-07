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


class GPT_Sentinel:
    def __init__(self, pretrained_model=True, n_samples=500):
        self.pretrained_model = pretrained_model
        self.n_samples = n_samples

    def main_func(self, data, model_collector, results_path, batch_size, pretrained_model_name):
        output = run_detect_experiment(data, pretrained_model_name, model_collector, batch_size, self.n_samples)
        with open(os.path.join(results_path, 'results.json'), 'w') as f:
            json.dump(output, f)

    def train_func(self, model_collector, cache_dir='./cache_dir', device='cuda', data='xsum', results_path='', pretrained_model_name='roberta-sentinel', learning_rate=5e-4, weight_decay=1e-3,
                   batch_size=512, max_epochs=15, max_sequence_length=512, epoch_size=None, loss_func=torch.nn.CrossEntropyLoss(), accumulaiton_steps=8):
        
        train_loader, validation_loader = load_datasets(cache_dir, model_collector.classification_tokenizer, batch_size, max_sequence_length, epoch_size)

        if pretrained_model_name == 'roberta-sentinel':
            for name, param in model_collector.classification_model.named_parameters():
                if 'roberta' in name:
                    param.requires_grad = False
            optimizer = AdamW(model_collector.classification_model.fc.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            optimizer = AdamW(model_collector.classification_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=max_epochs)
        epoch_loop = count(1) if max_epochs is None else range(1, max_epochs + 1)

        save_path = cache_dir + '/gpt_sentinel/'

        if not os.path.exists(save_path):
            os.mkdir(save_path)

        best_validation_accuracy = 0

        for epoch in epoch_loop:
            if pretrained_model_name == 'roberta-sentinel':
                train_loss, train_acc = train_roberta(model_collector.classification_model, optimizer, device, train_loader, loss_func, accumulaiton_steps)
                print(f'Training epoch {epoch} --- Training loss: {train_loss}, Training accuracy: {train_acc}', flush=True)
                val_loss, val_acc = validate_roberta(model_collector.classification_model, device, validation_loader, loss_func)
                print(f'Training epoch {epoch} --- Validation loss: {val_loss}, Validation accuracy: {val_acc}', flush=True)
            else:
                train_loss = train_t5(model_collector.classification_model, model_collector.classification_tokenizer, optimizer, device, train_loader, accumulaiton_steps)
                print(f'Training epoch {epoch} --- Training loss: {train_loss}', flush=True)
                val_loss, val_acc = validate_t5(model_collector.classification_model, model_collector.classification_tokenizer, device, validation_loader, loss_func)
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

# def train_roberta(model, optimizer, device, loader, loss_func):
#     model.train()

#     train_accuracy = 0
#     train_epoch_size = 0
#     train_loss = 0

#     for texts, masks, labels in loader:
#         texts, masks, labels = texts.to(device), masks.to(device), labels.to(device)
#         batch_size = texts.shape[0]

#         optimizer.zero_grad()
#         logits = model(texts, attention_mask=masks)
#         loss = loss_func(logits, labels)
#         loss.backward()
#         optimizer.step()

#         batch_accuracy = accuracy_sum(torch.softmax(logits, dim=-1), labels)
#         train_accuracy += batch_accuracy
#         train_epoch_size += batch_size
#         train_loss += loss.item() * batch_size

#     return train_loss / train_epoch_size, train_accuracy / train_epoch_size


def train_roberta(model, optimizer, device, loader, loss_func, accumulaiton_steps):
    model.train()

    train_accuracy = 0
    train_epoch_size = 0
    train_loss = 0

    for i, (texts, masks, labels) in enumerate(loader):
        texts, masks, labels = texts.to(device), masks.to(device), labels.to(device)
        batch_size = texts.shape[0]

        logits = model(texts, attention_mask=masks)
        loss = loss_func(logits, labels)
        loss = loss / accumulaiton_steps
        loss.backward()

        if (i+1) % accumulaiton_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        batch_accuracy = accuracy_sum(torch.softmax(logits, dim=-1), labels)
        train_accuracy += batch_accuracy
        train_epoch_size += batch_size
        train_loss += loss.item() * batch_size

    return train_loss / train_epoch_size, train_accuracy / train_epoch_size


# def train_t5(model, tokenizer, optimizer, device, loader, accumulaiton_steps):
#     model.train()

#     train_accuracy = 0
#     train_epoch_size = 0
#     train_loss = 0

#     positive_token_id = tokenizer("positive", return_tensors="pt")['input_ids'][0][0].item()  # = 1465, refers to ChatGPT-generated text 
#     positive_label = tokenizer.encode('positive </s>', return_tensors='pt')[0]
#     negative_label = tokenizer.encode('negative </s>', return_tensors='pt')[0]
#     negative_token_id = tokenizer("negative", return_tensors="pt")['input_ids'][0][0].item()  # = 2841, refers to human-written text 

#     for i, (texts, masks, labels) in enumerate(loader):
#         texts, masks, labels = texts.to(device), masks.to(device), labels.to(device)
#         batch_size = texts.shape[0]
#         t5_labels_list = []

#         optimizer.zero_grad()
#         for i in range(batch_size):
#             if labels[i] == 0:
#                 t5_labels_list.append(positive_label)
#             else:
#                 t5_labels_list.append(negative_label)
#         t5_labels = torch.stack(t5_labels_list).to(model.device)
#         decoder_input_ids = torch.tensor([tokenizer.pad_token_id] * batch_size).unsqueeze(-1).to(model.device)

#         outputs = model(texts, labels=labels, decoder_input_ids=decoder_input_ids)
#         loss = outputs['loss']
#         # logits = outputs['logits']
#         loss.backward()
#         optimizer.step()

#         # batch_accuracy = accuracy_sum(logits, labels)
#         # train_accuracy += batch_accuracy
#         train_epoch_size += batch_size
#         train_loss += loss.item() * batch_size

#     return train_loss / train_epoch_size

def train_t5(model, tokenizer, optimizer, device, loader, accumulaiton_steps):
    model.train()

    train_accuracy = 0
    train_epoch_size = 0
    train_loss = 0

    positive_token_id = tokenizer("positive", return_tensors="pt")['input_ids'][0][0].item()  # = 1465, refers to ChatGPT-generated text 
    positive_label = tokenizer.encode('positive', return_tensors='pt')[0]
    negative_label = tokenizer.encode('negative', return_tensors='pt')[0]
    negative_token_id = tokenizer("negative", return_tensors="pt")['input_ids'][0][0].item()  # = 2841, refers to human-written text 

    for i, (texts, masks, labels) in enumerate(loader):
        texts, masks, labels = texts.to(device), masks.to(device), labels.to(device)
        batch_size = texts.shape[0]
        t5_labels_list = []

        optimizer.zero_grad()
        for i in range(batch_size):
            if labels[i] == 0:
                t5_labels_list.append(positive_label)
            else:
                t5_labels_list.append(negative_label)
        t5_labels = torch.stack(t5_labels_list).to(model.device)
        decoder_input_ids = torch.tensor([tokenizer.pad_token_id] * batch_size).unsqueeze(-1).to(model.device)

        outputs = model(texts, labels=labels, decoder_input_ids=decoder_input_ids)
        loss = outputs['loss']
        loss = loss / accumulaiton_steps
        loss.backward()

        if (i+1) % accumulaiton_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        train_epoch_size += batch_size
        train_loss += loss.item() * batch_size

    return train_loss / train_epoch_size


def validate_roberta(model, device, loader, loss_func):
    model.eval()

    validation_accuracy = 0
    validation_epoch_size = 0
    validation_loss = 0

    with torch.no_grad():
        for texts, masks, labels in loader:
            texts, masks, labels = texts.to(device), masks.to(device), labels.to(device)
            batch_size = texts.shape[0]

            logits = model(texts, attention_mask=masks)
            loss = loss_func(logits, labels)

        batch_accuracy = accuracy_sum(torch.softmax(logits, dim=-1), labels)
        validation_accuracy += batch_accuracy
        validation_epoch_size += batch_size
        validation_loss += loss.item() * batch_size

    return validation_loss / validation_epoch_size, validation_accuracy / validation_epoch_size


def validate_t5(model, tokenizer, device, loader, loss_func):
    model.eval()

    validation_accuracy = 0
    validation_epoch_size = 0
    validation_loss = 0

    positive_token_id = tokenizer('positive', return_tensors='pt')['input_ids'][0][0].item()  # = 1465, refers to ChatGPT-generated text 
    negative_token_id = tokenizer('negative', return_tensors='pt')['input_ids'][0][0].item()  # = 2841, refers to human-written text 

    with torch.no_grad():
        for texts, masks, labels in loader:
            texts, masks, labels = texts.to(device), masks.to(device), labels.to(device)
            batch_size = texts.shape[0]

            decoder_input_ids = torch.tensor([tokenizer.pad_token_id] * batch_size).unsqueeze(-1).to(model.device)
            logits = model(input_ids=texts, decoder_input_ids=decoder_input_ids)
            logits = logits[0].squeeze(1) 
            selected_logits = logits[:, [positive_token_id, negative_token_id]] 
            logits = torch.softmax(selected_logits, dim=-1)
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
    real_dataset, fake_dataset = get_opengpttext_dataset(cache_dir)

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

