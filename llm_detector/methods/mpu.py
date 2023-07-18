# functions used to train DNN based binary classifiers for detection, for more information, see https://arxiv.org/pdf/2305.18149.pdf 
from itertools import count
import os
import torch
import torch.nn.functional as F
import torch.optim
import transformers
import params
import importlib
importlib.reload(params)
from utils import *

# --------- exp temp ---------
from exp_temp_all import *
# --------- exp temp ---------


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
            #print('All dynamic priors calculated...')


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


class MPU_classifier:
    def __init__(self, model_name, dataset_name, cache_dir, device):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.cache_dir = cache_dir
        self.device = device
        self.params = params.MPU_ROBERTA_PARAMS
        self.classification_model = transformers.RobertaForSequenceClassification.from_pretrained(self.model_name.replace('_', '-')[4:], cache_dir=self.cache_dir)
        self.classification_tokenizer = transformers.RobertaTokenizer.from_pretrained(self.model_name.replace('_', '-')[4:], cache_dir=self.cache_dir)
        self.classification_model = self.classification_model.to(self.device)


    def train_func(self, data):
        save_path = self.cache_dir + '/' + self.model_name 
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        # --------- exp temp ---------
        #train_loader, validation_loader = load_datasets(data, True, self.classification_tokenizer, self.params['batch_size'], params.OTHERS_PARAMS['padding_max_length'])
        train_loader, validation_loader = exp_loader(self.classification_tokenizer, batch_size=params.EXP_TEMP_PARAMS['batch_size'], name=self.dataset_name, prompt=params.EXP_TEMP_PARAMS['prompt'], padding_max_length=params.EXP_TEMP_PARAMS['padding_max_length'], train=True)
        # --------- exp temp ---------
        
        optimizer = torch.optim.AdamW(self.classification_model.parameters(), lr=self.params['learning_rate'], weight_decay=self.params['weight_decay'])

        best_validation_accuracy = 0

        for epoch in range(1, self.params['num_epochs'] + 1):
            train_loss, train_acc = train(self.classification_model, optimizer, self.device, train_loader, self.params['prior'], self.params['pu_type'], self.params['lamb'], self.params['len_thres'])
            print(f'Training epoch {epoch} --- Training loss: {train_loss}, Training accuracy: {train_acc}', flush=True)
            val_loss, val_acc = validate(self.classification_model, self.device, validation_loader)
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

        # load best checkpoint
        para = torch.load(os.path.join(save_path, self.dataset_name + '_best_model.pt'), map_location='cpu')
        self.classification_model.load_state_dict(para['model_state_dict'], strict=False)
        self.classification_model.eval()

        return self.classification_model, self.classification_tokenizer

