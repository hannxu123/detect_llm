import os
import transformers
import params
reload(params)



# load a public available pretrained model as detector
def load_public_pretrained_model(pretrained_model_name, cache_dir):
    pretrained_model_folder = cache_dir + '/pretrained_models' 

    if pretrained_model_name == 'gpt2_detector_roberta_base':
        download_path = 'https://openaipublic.azureedge.net/gpt-2/detector-models/v1/detector-base.pt'
        saved_path = pretrained_model_folder + '/detector-base.pt'
    elif pretrained_model_name == 'gpt2_detector_roberta_large':
        download_path = 'https://openaipublic.azureedge.net/gpt-2/detector-models/v1/detector-large.pt'
        saved_path = pretrained_model_folder + '/detector-large.pt'
    elif pretrained_model_name == 'roberta_sentinel':
        saved_path = pretrained_model_folder + '/roberta+mlp.base.0425.pt'
    elif pretrained_model_name == 't5_sentinel':
        saved_path = pretrained_model_folder + '/t5.small.0422.pt'

    if not os.path.exists(saved_path):
        try:
            import urllib.request
            urllib.request.urlretrieve(download_path, saved_path) 
        except:
            raise ValueError(f'Please download the pretrained model from https://drive.google.com/drive/folders/17IPZUaJ3Dd2LzsS8ezkelCfs5dMDOluD into {pretrained_model_folder} folder first')

    print(f'Loading checkpoint from {saved_path}', flush=True)
    para = torch.load(saved_path, map_location='cpu')

    if pretrained_model_name == 'gpt2_detector_roberta_base' or pretrained_model_name == 'gpt2_detector_roberta_large':
        classification_model = transformers.RobertaForSequenceClassification.from_pretrained(pretrained_model_name[14:].replace('_', '-'), cache_dir=cache_dir)
        classification_tokenizer = transformers.RobertaTokenizer.from_pretrained(pretrained_model_name[14:].replace('_', '-'), cache_dir=cache_dir)
        
        classification_model.load_state_dict(para['model_state_dict'], strict=False)
    elif pretrained_model_name == 'roberta_sentinel':
        classification_model = Roberta_Sentinel(cache_dir)
        classification_tokenizer = transformers.RobertaTokenizer.from_pretrained('roberta-base', cache_dir=cache_dir)
        
        classification_model.load_state_dict(para['model'], strict=False)
    elif pretrained_model_name == 't5_sentinel':
        from collections import OrderedDict
        classification_model = transformers.T5ForConditionalGeneration.from_pretrained('t5-small', cache_dir=cache_dir)
        classification_tokenizer = transformers.T5Tokenizer.from_pretrained('t5-small', cache_dir=cache_dir)

        new_state_dict = OrderedDict()
        for key, value in para['model'].items(): 
            name = key.replace('t5_model.', '')
            new_state_dict[name] = value

        classification_model.load_state_dict(new_state_dict, strict=False)

    classification_model.eval()

    return classification_model, classification_tokenizer


def load_availiable_tools(detection_tool_name):
    pass


# load an existing detector
def load_availiable_detector(detector_name, cache_dir):
    if detector_name in params.METHOD_PARAMS['supported_pretrained_models']:
        load_public_pretrained_model(detector_name, cache_dir)
    elif detector_name in params.METHOD_PARAMS['supported_tools']:
        load_availiable_tool(detector_name)






