import torch
import torch.nn as nn
import transformers


class model_collector:
    def __init__(self, base_model=None, base_tokenizer=None, mask_model=None, mask_tokenizer=None, classification_model=None, classification_tokenizer=None, 
                 device='cuda', openai_model=None, openai_key=None, do_top_k=False, top_k=40, do_top_p=False, top_p=0.96):
        self.base_model = base_model
        self.base_tokenizer = base_tokenizer
        self.mask_model = mask_model
        self.mask_tokenizer = mask_tokenizer
        self.classification_model = classification_model
        self.classification_tokenizer = classification_tokenizer
        self.device = device
        self.openai_model = openai_model
        self.openai_key = openai_key
        self.do_top_k = do_top_k
        self.top_k = top_k
        self.do_top_p = do_top_p
        self.top_p = top_p

    def load_base_model(self):
        try:
            self.mask_model.cpu()
        except:
            pass
        if self.openai_model is None:
            self.base_model.to(self.device)
        print(f'BASE model has moved to GPU', flush=True)

    def load_mask_model(self):
        if self.openai_model is None:
            self.base_model.cpu()

        self.mask_model.to(self.device)
        print(f'MASK model has moved to GPU', flush=True) 


    def load_classification_model(self):
        if self.openai_model is None and self.base_model is not None:
            self.base_model.cpu()

        self.classification_model.to(self.device)
        print(f'CLASSIFICATION model has moved to GPU', flush=True) 


# Roberta_Sentinel model used by gpt_sentinel method
class Roberta_Sentinel(nn.Module):
    def __init__(self, cache_dir): 
        super(Roberta_Sentinel, self).__init__()
        self.cache_dir = cache_dir 
        # load based model
        self.roberta = transformers.RobertaModel.from_pretrained('roberta-base', cache_dir=self.cache_dir)
        # define custom layers
        self.fc_1 = nn.Linear(768, 768)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.1)
        self.fc_2 = nn.Linear(768, 2)

    def forward(self, input_ids=None, attention_mask=None):
        # extract outputs from the body
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        outputs = outputs[0]
        # add custom layers
        outputs = self.fc_1(outputs[:,0,:].view(-1, 768))
        outputs = self.gelu(outputs)
        outputs = self.dropout(outputs)
        outputs = self.fc_2(outputs)

        return outputs

    @property
    def device(self):
        return next(self.parameters()).device


# Roberta based siamese network used by gpt_pat method
class SiameseNetwork(nn.Module):
    def __init__(self, cache_dir):
        super(SiameseNetwork, self).__init__()
        self.cache_dir = cache_dir 
        # load based model
        self.roberta = transformers.XLMRobertaModel.from_pretrained('xlm-roberta-base', cache_dir=self.cache_dir)
        self.cos_sim = nn.CosineSimilarity(dim=1)
        # add linear layers to compare between the features of the two images
        self.fc_1 = nn.Linear(768*2+1, 768)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.1)
        self.fc_2 = nn.Linear(768, 2)

    def forward_once(self, input_ids=None, attention_mask=None):
        output = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        #output = output.view(output.size()[0], -1)
        return output

    def forward(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2):
        # get two inputs' features
        output1 = self.forward_once(input_ids_1, attention_mask_1)[0] 
        output2 = self.forward_once(input_ids_2, attention_mask_2)[0]

        output1 = output1[:,0,:].view(-1, 768)
        output2 = output2[:,0,:].view(-1, 768)

        cos_output = self.cos_sim(output1, output2)
        cos_output = torch.unsqueeze(cos_output, dim=-1)

        outputs = torch.cat((output1, output2, cos_output), 1)
        outputs = self.fc_1(outputs[:,0,:].view(-1, 768))
        outputs = self.gelu(outputs)
        outputs = self.dropout(outputs)
        outputs = self.fc_2(ouputs)
        # # pass the out of the linear layers to sigmoid layer
        # outputs = self.sigmoid(outputs)
        
        return outputs

    @property
    def device(self):
        return next(self.parameters()).device


