# functions used to load/construct datasets
import pickle
import os
import random
import json
from dataset import construct_datasets, load_public_datasets
import params
import importlib
importlib.reload(params)

        
# load the cached dataset
def reload_cached_dataset(data_path):
    try:
        with open(data_path) as f:
            data = json.load(f)
        return data
    except:
        raise ValueError(f'Dataset does not exist.')


# load user customized dataset
def load_user_dataset(data_path):
    try:
        original_text = load_texts(os.path.join(writing_path, 'human_written_data.jsonl'))
        sampled_text = load_texts(os.path.join(writing_path, 'model_generated_data.jsonl'))
    except:
        raise ValueError(f'Cannot load data from {data_path}. Please process it first.')

    return original_text, sampled_text


# apply some post cleaning on collected data
def post_clean_dataset(original_text, sampled_text):
    original_text = list(dict.fromkeys(original_text))
    sampled_text = list(dict.fromkeys(sampled_text)) 

    original_text = [x.strip() for x in original_text]
    sampled_text = [x.strip() for x in sampled_text]

    original_text = [' '.join(x.split()) for x in original_text]
    sampled_text = [' '.join(x.split()) for x in sampled_text]

    random.shuffle(original_text)
    random.shuffle(sampled_text)

    final_data = {'original':original_text, 'sampled':sampled_text}

    return final_data


# load a dataset for detection
def load(full_dataset, human_dataset, synthetic_dataset, cache_dir, generation_model_name, generation_number, openai_model):
    data_folder = cache_dir + '/' + full_dataset
    if generation_model_name == '':
        data_path = data_folder + '/processed_final_data.json'
    else:
        data_path = data_folder + '/' + generation_model_name + '_processed_final_data.json'

    if os.path.exists(data_path):
        final_data = reload_cached_dataset(data_path)
    else:
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)

        if full_dataset in params.DATASET_PARAMS['supported_public_datasets']:
            original_text, sampled_text = load_public_datasets.load_public_dataset(full_dataset, synthetic_dataset, data_folder)
        elif full_dataset in params.DATASET_PARAMS['supported_model_generated_datasets']:
            original_text, sampled_text = construct_datasets.construct_predefined_dataset(full_dataset, data_folder, generation_model_name, generation_number, openai_model)
        else:
            original_text, sampled_text = load_user_dataset(data_folder)

        final_data = post_clean_dataset(original_text, sampled_text)

        with open(data_path, 'w') as f:
            json.dump(final_data, f)

    return final_data

