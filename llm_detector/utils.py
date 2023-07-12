import json
from typing import List
from transformers import PreTrainedTokenizer
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
from sklearn.metrics import roc_curve, precision_recall_curve, auc



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

        return tokenized_test['input_ids'], tokenized_test['attention_mask'], label


# load texts from a JSON file
def load_texts(data_file):
    with open(data_file) as f:
        data = [json.loads(line)['text'] for line in f]
    return data


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
        train_loader = DataLoader(train_dataset, batch_size, sampler=RandomSampler(train_dataset), num_workers=0)

        validation_dataset = EncodedDataset(original_valid, sampled_valid, tokenizer)
        validation_loader = DataLoader(validation_dataset, batch_size=1, sampler=RandomSampler(validation_dataset))

        return train_loader, validation_loader
    else:
        # test_dataset = EncodedDataset(original_test, sampled_test, tokenizer)
        # test_loader = DataLoader(test_dataset, batch_size=1)

        return original_test, sampled_test


def get_roc_metrics(real_preds, sample_preds):
    fpr, tpr, _ = roc_curve([0] * len(real_preds) + [1] * len(sample_preds), real_preds + sample_preds)
    roc_auc = auc(fpr, tpr)
    return float(roc_auc)


# def get_precision_recall_metrics(real_preds, sample_preds):
#     precision, recall, _ = precision_recall_curve([0] * len(real_preds) + [1] * len(sample_preds), real_preds + sample_preds)
#     pr_auc = auc(recall, precision)
#     return precision.tolist(), recall.tolist(), float(pr_auc)


def calculate_eval_metrics(label, pred_label, pred_posteriors):
    acc = accuracy_score(label, pred_label)
    precision = precision_score(label, pred_label)
    recall = recall_score(label, pred_label)
    f1 = f1_score(label, pred_label)
    auc = roc_auc_score(label, pred_posteriors)
    return acc, precision, recall, f1, auc


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
