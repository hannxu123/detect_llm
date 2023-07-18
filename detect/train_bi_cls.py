"""Training code for the detector model"""

import argparse
import torch
from torch import nn
from torch.optim import AdamW
import transformers
from transformers import tokenization_utils, RobertaTokenizer, RobertaForSequenceClassification
transformers.logging.set_verbosity_error()
import numpy as np
from detect.utils2 import balance_acc
from sklearn.metrics import f1_score
from detect.dataset2 import loader, loader_all
import random

def train(model, tokenizer, optimizer, device, loader):
    model.train()
    all_loss = []
    for i, dat in enumerate(loader):
        texts, labels = dat
        texts = list(texts)
        result = tokenizer(texts, return_tensors="pt", padding = 'max_length', max_length = 256, truncation=True)
        texts, masks, labels = result['input_ids'].to(device), result['attention_mask'].to(device), labels.to(device)
        aa = model(texts, labels=labels, attention_mask = masks)
        loss = aa['loss']
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        all_loss.append(loss.item())

def evaluate(model, tokenizer, device, loader):
    model.eval()
    m = nn.Softmax(dim = 1)
    with torch.no_grad():
        all_scores = []
        all_labels = []
        for i, dat in enumerate(loader):
            texts, labels = dat
            texts = list(texts)

            result = tokenizer(texts, return_tensors="pt", padding = 'max_length', max_length = 256, truncation=True)
            texts_encode, masks = result['input_ids'].to(device), result['attention_mask'].to(device),
            aa = model(texts_encode, attention_mask = masks)
            logits = aa['logits']
            score = m(logits)[:, 1]
            all_scores.append(score.cpu().numpy().flatten())
            all_labels.append(labels[0])

        all_scores_vec = np.concatenate(all_scores)
        all_labels = np.array(all_labels)
        acc = balance_acc(all_labels, all_scores_vec > 0.5)
        f1 = f1_score(all_labels, all_scores_vec > 0.5)
    return f1, acc

def run(batch_size=24,
        detect_model_name = 'base',
        learning_rate=2e-5,
        max_epoch = 3,
        train_name = 'reddit_eli5',
        prompt = 'p1',
        ):

    args = locals()
    device = 'cuda'
    all1 = []
    all2 = []
    all3 = []

    for seed in [100, 200, 300, 400, 500]:
        print('.......Now Random Seed ' + str(seed), flush=True)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        ## initiate RoBERTa model
        model_name = 'roberta-large' if detect_model_name == 'large' else 'roberta-base'
        tokenization_utils.logger.setLevel('ERROR')
        tokenizer = RobertaTokenizer.from_pretrained(model_name)
        model = RobertaForSequenceClassification.from_pretrained(model_name).to(device)

        # load the dataset
        ## dataset
        if (prompt == 'p1') or (prompt == 'p2') or (prompt == 'p3'):
            train_loader, _, valid_loader = loader(batch_size, prompt= prompt, train_name= train_name)
            _, test_loader1, _ = loader(batch_size, prompt= 'p1', train_name= train_name)
            _, test_loader2, _ = loader(batch_size, prompt= 'p2', train_name= train_name)
            _, test_loader3, _ = loader(batch_size, prompt= 'p3', train_name= train_name)
        else:
            train_loader, valid_loader, test_loader1, test_loader2, test_loader3 = \
                loader_all(batch_size, prompt= prompt, train_name= train_name)

        optimizer = AdamW(model.parameters(), lr=learning_rate)
        best_valid_f1 = -1
        f11 = 0
        f12 = 0
        f13 = 0

        for epoch in range(1, 1 + max_epoch):
            train(model, tokenizer, optimizer, device, train_loader)
            f1, acc = evaluate(model, tokenizer, device, valid_loader)
            if f1 > best_valid_f1:
                best_valid_f1 = f1
                print('.......')
                f11, acc = evaluate(model, tokenizer, device, test_loader1)
                f12, acc = evaluate(model, tokenizer, device, test_loader2)
                f13, acc = evaluate(model, tokenizer, device, test_loader3)
                print('Epoch ' +str(epoch), 'Current f1 ' + str(f1), 'Current best f1 ' + str(best_valid_f1), flush = True)
                print('Test Performance ', str(f11), str(f12), str(f13), flush= True)
            if best_valid_f1 > 0.9999:
                break
        all1.append(f11)
        all2.append(f12)
        all3.append(f13)

    print('#######################Final Performance ###############################')
    print('p1: ', np.mean(all1))
    print('p2: ', np.mean(all2))
    print('p3: ', np.mean(all3))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epoch', type=int, default= 10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--detect_model_name', type = str, default= 'base')
    parser.add_argument('--train_name', type = str, default= 'imdb')
    parser.add_argument('--prompt', type = str, default= 'p1')
    args = parser.parse_args()
    print('Training with RoBERTa')
    print(args)
    run(**vars(args))
