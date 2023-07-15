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
from detect.dataset2 import loader

def train(model, tokenizer, optimizer, device, loader):
    model.train()
    all_loss = []
    for i, dat in enumerate(loader):
        texts, labels = dat
        texts = list(texts)
        texts = tokenizer(texts, return_tensors="pt", padding = 'max_length', max_length = 64, truncation=True).input_ids
        texts, labels = texts.to(device), labels.to(device)
        aa = model(texts, labels=labels)
        loss = aa['loss']
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        all_loss.append(loss.item())

        if (i % 50 ==0) and (i > 0):
            print(np.mean(all_loss), flush = True)
            all_loss = []

def evaluate(model, tokenizer, device, loader):
    model.eval()
    m = nn.Softmax(dim = 1)
    with torch.no_grad():
        all_scores = []
        all_labels = []
        for i, dat in enumerate(loader):
            texts, labels = dat
            texts = list(texts)
            texts_encode = tokenizer(texts, return_tensors="pt", padding = 'max_length', max_length = 64, truncation=True).input_ids
            texts_encode = texts_encode.to(device)
            aa = model(texts_encode)
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
        seed = 100,
        train_name = 'reddit_eli5',
        prompt = 'p1',
        ):

    args = locals()
    device = 'cuda'
    torch.manual_seed(seed)
    np.random.seed(seed)

    ## initiate RoBERTa model
    model_name = 'roberta-large' if detect_model_name == 'large' else 'roberta-base'
    tokenization_utils.logger.setLevel('ERROR')
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForSequenceClassification.from_pretrained(model_name).to(device)

    # load the dataset
    ## dataset
    train_loader, _, valid_loader = loader(batch_size, name=train_name, prompt= prompt, verbose= True)
    _, test_loader1, _ = loader(batch_size, name=train_name, prompt= 'p1')
    _, test_loader2, _ = loader(batch_size, name=train_name, prompt= 'p2')
    _, test_loader3, _ = loader(batch_size, name=train_name, prompt= 'p3')


    optimizer = AdamW(model.parameters(), lr=learning_rate)
    best_valid_f1 = -1

    for epoch in range(1, 1 + max_epoch):
        print('Now Training Epoch ' + str(epoch), flush= True)
        train(model, tokenizer, optimizer, device, train_loader)
        print('Now Doing Validation Epoch ' + str(epoch), flush= True)
        f1, acc = evaluate(model, tokenizer, device, valid_loader)
        print('......')
        if f1 > best_valid_f1:
            best_valid_f1 = f1
            print('Now Evaluating on the Best')
            f1, acc = evaluate(model, tokenizer, device, test_loader1)
            print('P1 Test f1 score and accuracy ', f1, acc)
            f1, acc = evaluate(model, tokenizer, device, test_loader2)
            print('P2 Test f1 score and accuracy ', f1, acc)
            f1, acc = evaluate(model, tokenizer, device, test_loader3)
            print('P3 Test f1 score and accuracy ', f1, acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epoch', type=int, default= 10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=200)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--detect_model_name', type = str, default= 'base')
    parser.add_argument('--train_name', type = str, default= 'imdb')
    parser.add_argument('--prompt', type = str, default= 'p1')
    args = parser.parse_args()
    print('Training with RoBERTa')
    print(args)
    run(**vars(args))
