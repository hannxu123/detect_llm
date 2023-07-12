from itertools import count
import os
import torch
import torch.optim
import transformers
import params
import importlib
importlib.reload(params)
from utils import *


# train a roberta model (base or large)
def train_roberta(model, optimizer, device, loader):
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

    return train_loss / train_epoch_size, train_accuracy / train_epoch_size


# validate the trained roberta model
def validate_roberta(model, device, loader):
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

        batch_accuracy = accuracy_sum(logits, labels)
        validation_accuracy += batch_accuracy
        validation_epoch_size += batch_size
        validation_loss += loss.item() * batch_size

    return validation_loss / validation_epoch_size, validation_accuracy / validation_epoch_size


# train a roberta sentinel model 
def train_roberta_sentinel(model, optimizer, device, loader, loss_func, accumulaiton_steps):
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


# validate the trained roberta sentinel model
def validate_roberta_sentinel(model, device, loader, loss_func):
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


# train a t5 sentinel model
def train_t5_sentinel(model, tokenizer, optimizer, device, loader, accumulaiton_steps, positive_token_id, negative_token_id):
    model.train()

    train_accuracy = 0
    train_epoch_size = 0
    train_loss = 0

    for i, (texts, masks, labels) in enumerate(loader):
        texts, masks, labels = texts.to(device), masks.to(device), labels.to(device)
        batch_size = texts.shape[0]
        t5_labels_list = []

        optimizer.zero_grad()
        for i in range(batch_size):
            if labels[i] == 0:
                t5_labels_list.append(positive_token_id)
            else:
                t5_labels_list.append(negative_token_id)
        t5_labels = torch.LongTensor(t5_labels_list).to(model.device)
        decoder_input_ids = torch.tensor([tokenizer.pad_token_id] * batch_size).unsqueeze(-1).to(model.device)

        outputs = model(texts, labels=t5_labels, decoder_input_ids=decoder_input_ids)
        loss = outputs['loss']
        loss = loss / accumulaiton_steps
        loss.backward()

        if (i+1) % accumulaiton_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        train_epoch_size += batch_size
        train_loss += loss.item() * batch_size

    return train_loss / train_epoch_size, -1.0


# validate the trained t5 sentinel model
def validate_t5_sentinel(model, tokenizer, device, loader, loss_func, positive_token_id, negative_token_id):
    model.eval()

    validation_accuracy = 0
    validation_epoch_size = 0
    validation_loss = 0

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


class roberta_classifiers:
    def __init__(self, model_name, dataset_name, cache_dir, device):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.cache_dir = cache_dir
        self.device = device
        if self.model_name == 'roberta_base' or self.model_name == 'roberta_large':
            self.params = params.ROBERTA_PARAMS
            self.classification_model = transformers.RobertaForSequenceClassification.from_pretrained(self.model_name.replace('_', '-'), cache_dir=self.cache_dir)
            self.classification_tokenizer = transformers.RobertaTokenizer.from_pretrained(self.model_name.replace('_', '-'), cache_dir=self.cache_dir)
        elif self.model_name == 'roberta_sentinel':
            self.params = params.ROBERTA_SENTINEL_PARAMS
            self.classification_model = Roberta_Sentinel(self.cache_dir)
            self.classification_tokenizer = transformers.RobertaTokenizer.from_pretrained('roberta-base', cache_dir=self.cache_dir)
        else:
            self.params = params.T5_SENTINEL_PARAMS
            self.classification_model = transformers.T5ForConditionalGeneration.from_pretrained('t5-small', cache_dir=self.cache_dir)
            self.classification_tokenizer = transformers.T5Tokenizer.from_pretrained('t5-small', cache_dir=self.cache_dir)

        self.classification_model = self.classification_model.to(self.device)


    def train_func(self, data):
        save_path = self.cache_dir + '/' + self.model_name 
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        train_loader, validation_loader = load_datasets(data, True, self.classification_tokenizer, self.params['batch_size'], params.OTHERS_PARAMS['padding_max_length'])
        if self.params['optimizer'] == 'Adam':
            optimizer = torch.optim.Adam(self.classification_model.parameters(), lr=self.params['learning_rate'], weight_decay=self.params['weight_decay'])
        elif self.params['optimizer'] == 'AdamW':
            optimizer = torch.optim.AdamW(self.classification_model.parameters(), lr=self.params['learning_rate'], weight_decay=self.params['weight_decay'])
        else:
            raise ValueError(f'Optimizer {self.params["optimizer"]} has not been implemented yet. Please use "Adam" or "AdamW" as optimizer')

        best_validation_accuracy = 0

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.params['num_epochs'])

        for epoch in range(1, self.params['num_epochs'] + 1):
            # select a model to train
            if self.model_name == 'roberta_base' or self.model_name == 'roberta_large':
                train_loss, train_acc = train_roberta(self.classification_model, optimizer, self.device, train_loader)
                print(f'Training epoch {epoch} --- Training loss: {train_loss}, Training accuracy: {train_acc}', flush=True)
                val_loss, val_acc = validate_roberta(self.classification_model, self.device, validation_loader)
                print(f'Training epoch {epoch} --- Validation loss: {val_loss}, Validation accuracy: {val_acc}', flush=True)

            elif self.model_name == 'roberta_sentinel':
                if self.params['loss_func'] == 'cross_entropy':
                    loss_func = torch.nn.CrossEntropyLoss()
                else:
                    raise ValueError(f'Loss function {self.params["loss_func"]} has not been implemented yet. Please use "cross_entropy" as loss function')

                train_loss, train_acc = train_roberta_sentinel(self.classification_model, optimizer, self.device, train_loader, loss_func, self.params['accumulaiton_steps'])
                print(f'Training epoch {epoch} --- Training loss: {train_loss}, Training accuracy: {train_acc}', flush=True)
                val_loss, val_acc = validate_roberta_sentinel(self.classification_model, self.device, validation_loader, loss_func)
                print(f'Training epoch {epoch} --- Validation loss: {val_loss}, Validation accuracy: {val_acc}', flush=True)
            
            else:
                positive_token_id = self.classification_tokenizer('positive', return_tensors='pt')['input_ids'][0][0].item()  # = 1465, refers to ChatGPT-generated text 
                negative_token_id = self.classification_tokenizer('negative', return_tensors='pt')['input_ids'][0][0].item()  # = 2841, refers to human-written text 

                train_loss, train_acc = train_t5_sentinel(self.classification_model, self.classification_tokenizer, optimizer, self.device, train_loader, self.params['accumulaiton_steps'], positive_token_id, negative_token_id)
                print(f'Training epoch {epoch} --- Training loss: {train_loss}, Training accuracy: {train_acc}', flush=True)
                val_loss, val_acc = validate_t5_sentinel(self.classification_model, self.classification_tokenizer, self.device, validation_loader, loss_func, positive_token_id, negative_token_id)
                print(f'Training epoch {epoch} --- Validation loss: {val_loss}, Validation accuracy: {val_acc}', flush=True)
            

            if val_acc > best_validation_accuracy:
                best_validation_accuracy = val_acc

                model_to_save = self.classification_model.module if hasattr(self.classification_model, 'module') else self.classification_model
                torch.save(dict(
                        epoch=epoch,
                        model_state_dict=model_to_save.state_dict(),
                        optimizer_state_dict=optimizer.state_dict()
                    ),
                    os.path.join(save_path, self.dataset_name + '_best_model.pt')
                )
            scheduler.step()

        # load best checkpoint
        para = torch.load(save_path, map_location='cpu')
        self.classification_model.load_state_dict(para['model_state_dict'], strict=False)
        self.classification_model.eval()

        return self.classification_model, self.classification_tokenizer 
 
