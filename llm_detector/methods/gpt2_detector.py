import math
import numpy as np
import os
from typing import List
import json
from itertools import count
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torch.optim import Adam
from transformers import PreTrainedTokenizer
from utils import *


class GPT2_detector:
    def __init__(self, pretrained_model=True, n_samples=500):
        self.pretrained_model = pretrained_model
        self.n_samples = n_samples

    def main_func(self, data, model_collector, results_path, batch_size):
        output = run_detect_experiment(data, model_collector, batch_size, self.n_samples)
        with open(os.path.join(results_path, 'results.json'), 'w') as f:
            json.dump(output, f)

    def train_func(self, model_collector, seed=0, cache_dir='./cache_dir', device='cuda', real_dataset='webtext', fake_dataset='xl-1542M-nucleus', data='xsum', results_path='', learning_rate=2e-5, weight_decay=0,
                   batch_size=24, max_epochs=5, max_sequence_length=128, random_sequence_length=False, epoch_size=None, token_dropout=None):

        train_loader, validation_loader = load_datasets(cache_dir, real_dataset, fake_dataset, model_collector.classification_tokenizer, batch_size,
                                                    max_sequence_length, random_sequence_length, epoch_size, token_dropout, seed)

        optimizer = Adam(model_collector.classification_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        epoch_loop = count(1) if max_epochs is None else range(1, max_epochs + 1)

        save_path = cache_dir + '/gpt2_detector/'
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        # from torch.utils.tensorboard import SummaryWriter
        # writer = SummaryWriter(os.path.join(save_path, 'log'))
        best_validation_accuracy = 0

        for epoch in epoch_loop:
            train_loss, train_acc = train(model_collector.classification_model, optimizer, device, train_loader)
            print(f'Training epoch {epoch} --- Training loss: {train_loss}, Training accuracy: {train_acc}', flush=True)
            val_loss, val_acc  = validate(model_collector.classification_model, device, validation_loader)
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

            # combined_metrics = _all_reduce_dict({**validation_metrics, **train_metrics}, device)

            # combined_metrics["train/accuracy"] /= combined_metrics["train/epoch_size"]
            # combined_metrics["train/loss"] /= combined_metrics["train/epoch_size"]
            # combined_metrics["validation/accuracy"] /= combined_metrics["validation/epoch_size"]
            # combined_metrics["validation/loss"] /= combined_metrics["validation/epoch_size"]

            # for key, value in combined_metrics.items():
            #     writer.add_scalar(key, value, global_step=epoch)

            # if combined_metrics["validation/accuracy"] > best_validation_accuracy:
            #     best_validation_accuracy = combined_metrics["validation/accuracy"]

            #     model_to_save = model.module if hasattr(model, 'module') else model
            #     torch.save(dict(
            #             epoch=epoch,
            #             model_state_dict=model_to_save.state_dict(),
            #             optimizer_state_dict=optimizer.state_dict(),
            #             args=args
            #         ),
            #         os.path.join(save_path, 'best-model.pt')
            #     )

            output = run_detect_experiment(data, model_collector, batch_size, self.n_samples)
            with open(os.path.join(results_path, 'results.json'), 'w') as f:
                json.dump(output, f)


def train(model, optimizer, device, loader):
    model.train()

    train_accuracy = 0
    train_epoch_size = 0
    train_loss = 0

    for texts, masks, labels in loader:
        texts, masks, labels = texts.to(device), masks.to(device), labels.to(device)
        batch_size = texts.shape[0]

        optimizer.zero_grad()
        outputs = model(texts, attention_mask=masks, labels=labels)
        loss = outputs['loss']
        logits = outputs['logits']
        loss.backward()
        optimizer.step()

        batch_accuracy = accuracy_sum(logits, labels)
        train_accuracy += batch_accuracy
        train_epoch_size += batch_size
        train_loss += loss.item() * batch_size

        #loop.set_postfix(loss=loss.item(), acc=train_accuracy / train_epoch_size)

    # return {
    #     "train/accuracy": train_accuracy,
    #     "train/epoch_size": train_epoch_size,
    #     "train/loss": train_loss
    # }
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

            outputs = model(texts, attention_mask=masks, labels=labels)
            loss = outputs['loss']
            logits = outputs['logits']
            # losses.append(loss)
            # logit_votes.append(logits)

        batch_accuracy = accuracy_sum(logits, labels)
        validation_accuracy += batch_accuracy
        validation_epoch_size += batch_size
        validation_loss += loss.item() * batch_size

        #loop.set_postfix(loss=loss.item(), acc=validation_accuracy / validation_epoch_size)

    # return {
    #     "validation/accuracy": validation_accuracy,
    #     "validation/epoch_size": validation_epoch_size,
    #     "validation/loss": validation_loss
    # }
    return validation_loss / validation_epoch_size, validation_accuracy / validation_epoch_size


def accuracy_sum(logits, labels):
    if list(logits.shape) == list(labels.shape) + [2]:
        # 2-d outputs
        classification = (logits[..., 0] < logits[..., 1]).long().flatten()
    else:
        classification = (logits > 0).long().flatten()
    assert classification.shape == labels.shape
    return (classification == labels).float().sum().item()


# def _all_reduce_dict(d, device):
#     # wrap in tensor and use reduce to gpu0 tensor
#     output_d = {}
#     for (key, value) in sorted(d.items()):
#         tensor_input = torch.tensor([[value]]).to(device)
#         torch.distributed.all_reduce(tensor_input)
#         output_d[key] = tensor_input.item()
#     return output_d


def load_datasets(cache_dir, real_dataset, fake_dataset, tokenizer, batch_size, max_sequence_length, random_sequence_length, epoch_size=None, token_dropout=None, seed=None):
    if fake_dataset == 'TWO':
        get_gpt2_detector_dataset(real_dataset, 'xl-1542M', 'xl-1542M-nucleus', cache_dir=cache_dir, train_model=True)
    elif fake_dataset == 'THREE':
        get_gpt2_detector_dataset(real_dataset, 'xl-1542M', 'xl-1542M-k40', 'xl-1542M-nucleus', cache_dir=cache_dir, train_model=True)
    else:
        get_gpt2_detector_dataset(real_dataset, fake_dataset, cache_dir=cache_dir, train_model=True)

    real_corpus = Corpus(real_dataset, cache_dir=cache_dir)

    if fake_dataset == "TWO":
        real_train, real_valid = real_corpus.train * 2, real_corpus.valid * 2
        fake_corpora = [Corpus(name, cache_dir=cache_dir) for name in ['xl-1542M', 'xl-1542M-nucleus']]
        fake_train = sum([corpus.train for corpus in fake_corpora], [])
        fake_valid = sum([corpus.valid for corpus in fake_corpora], [])
    elif fake_dataset == "THREE":
        real_train, real_valid = real_corpus.train * 3, real_corpus.valid * 3
        fake_corpora = [Corpus(name, cache_dir=cache_dir) for name in ['xl-1542M', 'xl-1542M-k40', 'xl-1542M-nucleus']]
        fake_train = sum([corpus.train for corpus in fake_corpora], [])
        fake_valid = sum([corpus.valid for corpus in fake_corpora], [])
    else:
        fake_corpus = Corpus(fake_dataset, cache_dir=cache_dir)

        real_train, real_valid = real_corpus.train, real_corpus.valid
        fake_train, fake_valid = fake_corpus.train, fake_corpus.valid

    min_sequence_length = 10 if random_sequence_length else None
    train_dataset = EncodedDataset(real_train, fake_train, tokenizer, max_sequence_length, min_sequence_length, epoch_size, token_dropout, seed)
    train_loader = DataLoader(train_dataset, batch_size, sampler=RandomSampler(train_dataset), num_workers=0)

    validation_dataset = EncodedDataset(real_valid, fake_valid, tokenizer)
    validation_loader = DataLoader(validation_dataset, batch_size=1, sampler=RandomSampler(validation_dataset))

    return train_loader, validation_loader


def run_detect_experiment(data, model_collector, batch_size, n_samples=500):
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
            pred_probs = torch.softmax(pred_probs[0], dim=-1)
            # binary output format [fake_pred, real_pred], take probability of model-generated text (fake) as positive
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
            pred_probs = torch.softmax(pred_probs[0], dim=-1)

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
    def __init__(self, name, cache_dir, skip_train=False):
        # download(name, cache_dir=cache_dir)
        self.name = name
        self.train = load_texts(f'{cache_dir}/{name}.train.jsonl', expected_size=250000) if not skip_train else None
        self.valid = load_texts(f'{cache_dir}/{name}.valid.jsonl', expected_size=5000)
        self.test = load_texts(f'{cache_dir}/{name}.test.jsonl', expected_size=5000)


class EncodedDataset(Dataset):
    def __init__(self, real_texts: List[str], fake_texts: List[str], tokenizer: PreTrainedTokenizer,
                 max_sequence_length: int = None, min_sequence_length: int = None, epoch_size: int = None,
                 token_dropout: float = None, seed: int = None):
        self.real_texts = real_texts
        self.fake_texts = fake_texts
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
        self.min_sequence_length = min_sequence_length
        self.epoch_size = epoch_size
        self.token_dropout = token_dropout
        self.random = np.random.RandomState(seed)

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

        tokens = self.tokenizer.encode(text)
        tokens = tokens[1:len(tokens)-1]

        if self.max_sequence_length is None:
            tokens = tokens[:self.tokenizer.model_max_length - 2]
        else:
            output_length = min(len(tokens), self.max_sequence_length)
            if self.min_sequence_length:
                output_length = self.random.randint(min(self.min_sequence_length, len(tokens)), output_length + 1)
            start_index = 0 if len(tokens) <= output_length else self.random.randint(0, len(tokens) - output_length + 1)
            end_index = start_index + output_length
            tokens = tokens[start_index:end_index]

        if self.token_dropout:
            dropout_mask = self.random.binomial(1, self.token_dropout, len(tokens)).astype(np.bool)
            tokens = np.array(tokens)
            tokens[dropout_mask] = self.tokenizer.unk_token_id
            tokens = tokens.tolist()

        if self.max_sequence_length is None or len(tokens) == self.max_sequence_length:
            mask = torch.ones(len(tokens) + 2)
            return torch.tensor([self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]), mask, label

        padding = [self.tokenizer.pad_token_id] * (self.max_sequence_length - len(tokens))
        tokens = torch.tensor([self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id] + padding)
        mask = torch.ones(tokens.shape[0])
        mask[-len(padding):] = 0
        return tokens, mask, label



