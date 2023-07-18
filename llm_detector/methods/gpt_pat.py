from itertools import count
import os
import torch
import torch.optim
import transformers
import params
from models import Roberta_Sentinel
import importlib
importlib.reload(params)
from utils import *




class GPT_Pat:
    def __init__(self, model_name, dataset_name, cache_dir, device):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.cache_dir = cache_dir
        self.device = device
        self.classification_model = SiameseNetwork(self.cache_dir)
        self.classification_tokenizer = transformers.RobertaTokenizer.from_pretrained('roberta-base', cache_dir=self.cache_dir)
        self.params = params.GPT_PAT_PARAMS
        self.classification_model = self.classification_model.to(self.device)


    def train_func(self, data):
        save_path = self.cache_dir + '/' + self.model_name 
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        train_loader, validation_loader = load_datasets(data, True, self.classification_tokenizer, self.params['batch_size'], params.OTHERS_PARAMS['padding_max_length'])
        


openai.api_key = ''


class GPT_Pat:
    def __init__(self, pretrained_model=False, n_samples=5000):
        self.pretrained_model = pretrained_model
        self.n_samples = n_samples

    def main_func(self, data, model_collector, results_path):
        pass

    def train_func(self, data, model_collector, cache_dir='./cache_dir', device='cuda', results_path='', learning_rate=5e-5, weight_decay=1e-3,
               batch_size=32, max_epochs=10, max_sequence_length=512, epoch_size=None, loss_func=torch.nn.CrossEntropyLoss()):
        
        #if not os.path.isdir(cache_dir + '/gpt_pat_data/'):
        original_text = data['original']
        sampled_text = data['sampled']

        create_text(original_text, 1000, './cache_dir', self.n_samples, 'original_text')
        create_text(sampled_text, 1000, './cache_dir', self.n_samples, 'sampled_text')

        train_loader, validation_loader, test_data = load_datasets(cache_dir, model_collector.classification_tokenizer, batch_size, max_sequence_length, epoch_size)

        optimizer = Adam(model_collector.classification_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        epoch_loop = count(1) if max_epochs is None else range(1, max_epochs + 1)

        save_path = cache_dir + '/gpt_pat/'

        if not os.path.exists(save_path):
            os.mkdir(save_path)

        best_validation_accuracy = 0

        for epoch in epoch_loop:
            train_loss, train_acc = train(model_collector.classification_model, optimizer, device, train_loader, loss_func)
            print(f'Training epoch {epoch} --- Training loss: {train_loss}, Training accuracy: {train_acc}', flush=True)
            val_loss, val_acc = validate(model_collector.classification_model, device, validation_loader, loss_func)
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

            output = run_detect_experiment(test_data, model_collector, batch_size, self.n_samples)
            with open(os.path.join(results_path, 'results.json'), 'w') as f:
                json.dump(output, f)


def run_gpt(prompt):
    completions = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        # max_tokens=1024,
        # n=2,
        # stop=None,
        temperature=0.2,
        messages=[{'role': 'user', 'content': prompt}]
    )

    message = completions['choices'][0]['message']['content']
    return message.strip()


def create_text(data, start_index, cache_dir, num_sample, category):
    nums = min(len(data) - start_index, num_sample)

    json_list = []
    folder_path = cache_dir + '/gpt_pat_data/'
    json_file_name = folder_path + 'hc3_' + category + '.jsonl'

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    end_index = start_index + nums
    count = 0
    i = start_index
    while i < end_index:
        cur_text = data[i]
        len_text = len(cur_text.split())

        qg_prompt = 'I want you to play the role of the questioner. I will type an answer in English, and you will ask me a question based on the answer in the same language. Don’t write any explanations or other text, just give me the question. ' + cur_text
        try:
            questions = run_gpt(qg_prompt)
            re_answer_prompt = questions + f' Answer in {len_text} words or less.'
            try:
                re_answer_text = run_gpt(re_answer_prompt)
                
                json_content = {
                    'input_text': cur_text,
                    'model_gene_text': re_answer_text,
                }
                json_list.append(json_content)

                print(f'No.{i+1} sample has been generated by model')
                i += 1
            except:
                count += 1
                print(f'Cannot get model answer for No.{i+1} sample, try again')
                if os.path.exists(json_file_name):
                    json_file_signal = 'a'
                else:
                    json_file_signal = 'w'

                with open(json_file_name, json_file_signal, encoding='utf-8') as json_file:
                    for entry in json_list:
                        json.dump(entry, json_file)
                        json_file.write('\n')
        except:
            count += 1
            print(f'Cannot get model response for No.{i+1} sample, try again')

            if os.path.exists(json_file_name):
                json_file_signal = 'a'
            else:
                json_file_signal = 'w'

            with open(json_file_name, json_file_signal, encoding='utf-8') as json_file:
                for entry in json_list:
                    json.dump(entry, json_file)
                    json_file.write('\n')

            json_list = []
            time.sleep(60*count)

    if os.path.exists(json_file_name):
        json_file_signal = 'a'
    else:
        json_file_signal = 'w'

    with open(json_file_name, json_file_signal, encoding='utf-8') as json_file:
        for entry in json_list:
            json.dump(entry, json_file)
            json_file.write('\n')


def train(model, optimizer, device, loader, loss_func):
    model.train()

    train_accuracy = 0
    train_epoch_size = 0
    train_loss = 0

    for texts, masks, labels in loader:
        orig_texts, gene_texts = texts[0].to(device), texts[1].to(device)
        orig_masks, gene_masks = masks[0].to(device), masks[1].to(device)
        labels = labels.to(device)
        batch_size = orig_texts.shape[0]

        optimizer.zero_grad()
        logits = model(orig_texts, orig_masks, gene_texts, gene_masks)
        loss = loss_func(logits, labels)
        loss.backward()
        optimizer.step()

        batch_accuracy = accuracy_sum(torch.softmax(logits, dim=-1), labels)
        train_accuracy += batch_accuracy
        train_epoch_size += batch_size
        train_loss += loss.item() * batch_size

    return train_loss / train_epoch_size, train_accuracy / train_epoch_size


def validate(model, device, loader, loss_func):
    model.eval()

    validation_accuracy = 0
    validation_epoch_size = 0
    validation_loss = 0

    with torch.no_grad():
        for texts, masks, labels in loader:
            orig_texts, gene_texts = texts[0].to(device), texts[1].to(device)
            orig_masks, gene_masks = masks[0].to(device), masks[1].to(device)
            labels = labels.to(device)
            batch_size = orig_texts.shape[0]

            logits = model(orig_texts, orig_masks, gene_texts, gene_masks)
            loss = loss_func(logits, labels)

        batch_accuracy = accuracy_sum(logits, labels)
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
    folder_path = cache_dir + '/gpt_pat_data/'

    with open(folder_path + 'hc3_original_text.jsonl') as f:
        real_dataset_orig = [json.loads(line)['input_text'] for line in f]
    with open(folder_path + 'hc3_original_text.jsonl') as f:
        real_dataset_gene = [json.loads(line)['model_gene_text'] for line in f]

    with open(folder_path + 'hc3_sampled_text.jsonl') as f:
        fake_dataset_orig = [json.loads(line)['input_text'] for line in f]
    with open(folder_path + 'hc3_sampled_text.jsonl') as f:
        fake_dataset_gene = [json.loads(line)['model_gene_text'] for line in f]

    real_corpus_orig = Corpus(real_dataset_orig, cache_dir=cache_dir)
    real_corpus_gene = Corpus(real_dataset_gene, cache_dir=cache_dir)
    fake_corpus_orig = Corpus(fake_dataset_orig, cache_dir=cache_dir)
    fake_corpus_gene = Corpus(fake_dataset_gene, cache_dir=cache_dir)

    real_train_orig, real_valid_orig, real_test_orig = real_corpus_orig.train, real_corpus_orig.valid, real_corpus_orig.test
    real_train_gene, real_valid_gene, real_test_gene = real_corpus_gene.train, real_corpus_gene.valid, real_corpus_gene.test
    fake_train_orig, fake_valid_orig, fake_test_orig = fake_corpus_orig.train, fake_corpus_orig.valid, fake_corpus_orig.test
    fake_train_gene, fake_valid_gene, fake_test_gene = fake_corpus_gene.train, fake_corpus_gene.valid, fake_corpus_gene.test

    train_dataset = EncodedDataset(real_train_orig, real_train_gene, fake_train_orig, fake_train_gene, tokenizer, max_sequence_length, epoch_size)
    train_loader = DataLoader(train_dataset, batch_size, sampler=RandomSampler(train_dataset), num_workers=0)

    validation_dataset = EncodedDataset(real_valid_orig, real_valid_gene, fake_valid_orig, fake_valid_gene, tokenizer)
    validation_loader = DataLoader(validation_dataset, batch_size=1, sampler=RandomSampler(validation_dataset))

    return train_loader, validation_loader, (real_test_orig, real_test_gene, fake_test_orig, fake_test_gene)


def run_detect_experiment(data, model_collector, batch_size, n_samples=500):
    results = []
    real_test_orig, real_test_gene, fake_test_orig, fake_test_gene = data[0], data[1], data[2], data[3]

    with torch.no_grad():
        # get predictions for original text
        real_preds = []
        for batch in range(math.ceil(len(real_test_orig) / batch_size)): 
            try:
                batch_real_orig = real_test_orig[batch * batch_size:(batch + 1) * batch_size]
                batch_real_gene = real_test_gene[batch * batch_size:(batch + 1) * batch_size]
            except:
                batch_real_orig = real_test_orig[batch * batch_size:len(real_test_orig)]
                batch_real_gene = real_test_gene[batch * batch_size:len(real_test_orig)]

            tokenized_batch_real_orig = model_collector.classification_tokenizer(batch_real_orig, padding=True, truncation=True, max_length=512, return_tensors="pt").to(model_collector.classification_model.device)
            tokenized_batch_real_gene = model_collector.classification_tokenizer(batch_real_gene, padding=True, truncation=True, max_length=512, return_tensors="pt").to(model_collector.classification_model.device)
 
            pred_probs = model_collector.classification_model(input_ids_1=tokenized_batch_real_orig['input_ids'], attention_mask_1=tokenized_batch_real_orig['attention_mask'], input_ids_2=tokenized_batch_real_gene['input_ids'], attention_mask_2=tokenized_batch_real_gene['attention_mask'])
            pred_probs = torch.softmax(pred_probs, dim=-1)
            # binary output format [fake_pred, real_pred], take probability of model-generated text (fake) as positive
            real_preds.extend(pred_probs[:,0].detach().cpu().numpy().tolist())
        
        # get predictions for sampled text
        fake_preds = []
        for batch in range(math.ceil(len(fake_test_orig) / batch_size)):
            try:
                batch_fake_orig = fake_test_orig[batch * batch_size:(batch + 1) * batch_size]
                batch_fake_gene = fake_test_gene[batch * batch_size:(batch + 1) * batch_size]
            except:
                batch_fake_orig = fake_test_orig[batch * batch_size:len(fake_test_orig)]
                batch_fake_gene = fake_test_gene[batch * batch_size:len(fake_test_orig)]

            tokenized_batch_fake_orig = model_collector.classification_tokenizer(batch_fake_orig, padding=True, truncation=True, max_length=512, return_tensors="pt").to(model_collector.classification_model.device)
            tokenized_batch_fake_gene = model_collector.classification_tokenizer(batch_fake_gene, padding=True, truncation=True, max_length=512, return_tensors="pt").to(model_collector.classification_model.device)
            pred_probs = model_collector.classification_model(input_ids_1=tokenized_batch_fake_orig['input_ids'], attention_mask_1=tokenized_batch_fake_orig['attention_mask'], input_ids_2=tokenized_batch_fake_gene['input_ids'], attention_mask_2=tokenized_batch_fake_gene['attention_mask'])
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

        tokens_orig = self.tokenizer.encode(text[0], truncation=True, max_length=self.max_sequence_length)
        tokens_gene = self.tokenizer.encode(text[1], truncation=True, max_length=self.max_sequence_length)

        padding_orig = [self.tokenizer.pad_token_id] * (self.max_sequence_length - len(tokens_orig))
        padding_gene = [self.tokenizer.pad_token_id] * (self.max_sequence_length - len(tokens_gene))
        tokens_orig = torch.tensor(tokens_orig + padding_orig)
        tokens_gene = torch.tensor(tokens_gene + padding_gene)
        mask_orig = torch.ones(tokens_orig.shape[0])
        mask_gene = torch.ones(tokens_gene.shape[0])
        mask_orig[-len(padding_orig):] = 0
        mask_gene[-len(padding_gene):] = 0
        return (tokens_orig, tokens_gene), (mask_orig, mask_gene), label
