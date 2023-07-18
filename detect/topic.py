"""Training code for the detector model"""

import argparse
import torch
from torch import nn
from torch.optim import AdamW
import transformers
from transformers import tokenization_utils, RobertaTokenizer, RobertaForSequenceClassification
transformers.logging.set_verbosity_error()
import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader
import pickle

def train(model, tokenizer, optimizer, device, loader):
    model.train()
    all_loss = []
    for i, dat in enumerate(loader):
        texts, labels = dat['text'], dat['label']
        texts = list(texts)
        result = tokenizer(texts, return_tensors="pt", padding = 'max_length', max_length = 256, truncation=True)
        texts, masks, labels = result['input_ids'].to(device), result['attention_mask'].to(device), labels.to(device)
        aa = model(texts, labels=labels, attention_mask = masks)
        loss = aa['loss']
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        all_loss.append(loss.item())

        if i % 300 == 0:
            print('step ' +str(i), loss.item(), flush= True)

def evaluate(model, tokenizer, device, loader):
    model.eval()
    m = nn.Softmax(dim = 1)
    all_label = []
    all_pred = []
    with torch.no_grad():
        for i, dat in enumerate(loader):
            texts, labels = dat['text'], dat['label']
            texts = list(texts)
            result = tokenizer(texts, return_tensors="pt", padding = 'max_length', max_length = 256, truncation=True)
            texts_encode, masks = result['input_ids'].to(device), result['attention_mask'].to(device),
            aa = model(texts_encode, attention_mask = masks)
            logits = m(aa['logits'])

            pred = torch.argmax(logits)
            # confidence = torch.max(logits).item()
            all_label.append(labels[0])
            all_pred.append(pred.item())
            if i > 500:
                break

    all_label = np.array(all_label)
    all_pred = np.array(all_pred)
    accuracy = np.sum(all_label == all_pred) / all_label.shape[0]
    print('Test Accuracy is ' +str(accuracy), flush= True)
    return accuracy


def evaluate2(model, tokenizer, device):
    test_set = load_dataset("xsum")['train']
    test_loader = DataLoader(test_set, 1, shuffle=True, num_workers=0)
    model.eval()
    m = nn.Softmax(dim=1)

    all0 = []
    all1 = []
    all2 = []
    all3 = []
    with torch.no_grad():
        for i, dat in enumerate(test_loader):
            texts = dat['summary']
            texts = list(texts)
            result = tokenizer(texts, return_tensors="pt", padding='max_length', max_length=256, truncation=True)
            texts_encode, masks = result['input_ids'].to(device), result['attention_mask'].to(device),
            aa = model(texts_encode, attention_mask=masks)
            logits = m(aa['logits'])

            pred = torch.argmax(logits)
            confidence = torch.max(logits).item()

            if (confidence > 0.8) & (pred == 0):
                all0.append(texts[0])
            if (confidence > 0.8) & (pred == 1):
                all1.append(texts[0])
            if (confidence > 0.8) & (pred == 2):
                all2.append(texts[0])
            if (confidence > 0.8) & (pred == 3):
                all3.append(texts[0])
            if i % 1000 == 0:
                print('Processing sample ' +str(i), flush= True)

    print(len(all0), len(all1), len(all2), len(all3))
    dd = {'world': all0, 'sports': all1, 'business': all2, 'sci': all3}
    with open('data_save/real_news', "wb") as fp1:  # Pickling
        pickle.dump(dd, fp1)


def run(batch_size=24,
        detect_model_name = 'base',
        learning_rate=2e-5,
        max_epoch = 3,
        ):

    args = locals()
    device = 'cuda'
    all1 = []
    all2 = []
    all3 = []

    ## initiate RoBERTa model
    model_name = 'roberta-large' if detect_model_name == 'large' else 'roberta-base'
    tokenization_utils.logger.setLevel('ERROR')

    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels = 4).to(device)

    # load the dataset
    ## dataset
    data_set = load_dataset("ag_news")
    train_set = data_set['train']
    test_set = data_set['test']
    train_loader = DataLoader(train_set, batch_size, shuffle=True, num_workers=1)
    test_loader = DataLoader(test_set, 1, shuffle=True, num_workers=0)
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(1, 1 + max_epoch):
        train(model, tokenizer, optimizer, device, train_loader)
        print('Now doing evaluation')
        acc = evaluate(model, tokenizer, device, test_loader)
    print('#######################################')
    print('Now checking the XSUM dataset')
    evaluate2(model, tokenizer, device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epoch', type=int, default= 3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--detect_model_name', type = str, default= 'base')
    args = parser.parse_args()
    print('Training with RoBERTa')
    print(args)
    run(**vars(args))
