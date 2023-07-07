import argparse

parser = argparse.ArgumentParser(description='Detect LLM generated texts')

parser.add_argument('--seed', type=int, default=0, help='seed')
parser.add_argument('--detect_method', default='gpt_sentinel', type=str, help='name of detection method')
parser.add_argument('--target_text', default='', type=str, help='text for detection')
parser.add_argument('--human_dataset_name', type=str, default='xsum', help='name of the human written dataset')
parser.add_argument('--synthetic_dataset_name', type=str, default='', help='name of the model generated dataset')
parser.add_argument('--dataset_split', type=str, default='train', help='split of the dataset when downloading from huggingface')
parser.add_argument('--key_name', type=str, default='', help='key name of the dataset when downloading from huggingface')
parser.add_argument('--user_data_path', type=str, default='', help='path of user preprocessed dataset')
parser.add_argument('--num_texts', type=int, default=500, help='number of text samples used in experiments')
parser.add_argument('--text_maximum_lenth', type=int, default=-1, help='maximum length of text used in experiments')
parser.add_argument('--text_minimum_lenth', type=int, default=-1, help='minimum length of text used in experiments')

parser.add_argument('--base_model_name', type=str, default='gpt2-medium', help='model to generate model written texts based on huggingface datasets')
parser.add_argument('--openai_model', type=str, default=None, help='whether use openai model as base model to generate model written texts ')
parser.add_argument('--openai_key', type=str, default='', help='key for openai api')
parser.add_argument('--device', default='cuda', type=str, help='device used to execute')





# parser.add_argument('--customized_dataset', type=str, default='', help='whether using customized dataset for detection')

# parser.add_argument('--predefined_dataset', type=str, default='', help='whether using customized dataset for detection')


# parser.add_argument('--dataset', default='xsum', type=str, help='dataset used for model training and evaluation')











parser.add_argument('--batch_size', type=int, default=50)




parser.add_argument('--cache_dir', type=str, default='./cache_dir', help='path of the folder to save cache')
parser.add_argument('--output_to_log', default='', help='whether saving logs into a file')


parser.add_argument('--pretrained_model', type=str, default='yes', help='whether using a pretrained detector in gpt2_detector')
parser.add_argument('--pretrained_model_name', type=str, default='roberta-base', help='name of model used in gpt2_detector')

parser.add_argument('--local', default='', type=str, help='the gpu number used on developing node.')

args = parser.parse_args()

import os
if args.local != '':
    os.environ['CUDA_VISIBLE_DEVICES'] = args.local

import numpy as np
import torch
import random
import sys
import json
import subprocess
import transformers
from utils_new import *
from process_datasets import *
from models import *
from detect_gpt import *
from gpt2_detector import *
from gltr import *
from gpt_sentinel import *


def main():
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # cudnn.deterministic = True

    if args.synthetic_dataset_name == '':
        synthetic_dataset_name = 'SAME'
    else:
        synthetic_dataset_name = args.synthetic_dataset_name

    if args.target_text == '':
        print(f'Use DATASET {args.human_dataset_name} as human-written data and DATASET {synthetic_dataset_name} as model-generated data to detect', flush=True)
    print(f'Seed number is {args.seed}', flush=True)

    device = args.device
    print('Use GPU to run', flush=True) if device == 'cuda' else print('Use CPU to run', flush=True)

    if args.target_text != '':
        pass
    else:
        # create a cache folder to save models and datasets
        if not os.path.exists(args.cache_dir):
            os.makedirs(args.cache_dir)

        data_path = args.cache_dir + f'/{args.human_dataset_name}_{synthetic_dataset_name}'

        results_path = f'./detection_results/{args.detect_method}/{args.human_dataset_name}_{synthetic_dataset_name}'
        if not os.path.exists(results_path):
            os.makedirs(results_path)
        print(f'Saving results to absolute path: {os.path.abspath(results_path)}')

        if args.output_to_log:
            log_path = os.path.join(results_path, 'outputs.log')
            log = open(log_path, 'w')
            sys.stdout = log
            print(f'Saving logs to absolute path: {os.path.abspath(log_path)}')

        if os.path.exists(os.path.join(data_path, 'experiment_data.json')):
            # load a cached dataset
            final_data = reload_cached_dataset(data_path)
        else:
            if not os.path.exists(data_path):
                os.makedirs(data_path)

            if synthetic_dataset_name == 'SAME':
                if args.openai_model is None:
                    base_model_name = args.base_model_name.replace('/', '_')
                else:
                    base_model_name = "openai-" + args.openai_model.replace('/', '_')

                if args.key_name == '':
                    # construct a new model-written dataset based on a predefined human-written dataset
                    data = construct_predefined_dataset(args.human_dataset_name, data_path)
                else:
                    # construct a new model-written dataset by using a human-written dataset from HF
                    data = construct_customized_hf_dataset(args.human_dataset_name, args.dataset_split, args.key_name, data_path)

                data = data[:args.num_texts]

                # # keep only examples with <= 512 tokens according to mask_tokenizer
                # # this step has the extra effect of removing examples with low-quality/garbage content
                # data = cur_dataset.data[:5000]
                # tokenized_data = preproc_tokenizer(data)
                # data = [x for x, y in zip(data, tokenized_data["input_ids"]) if len(y) <= 512]

                # # print stats about remainining data
                # print(f"Total number of samples: {len(data)}", flush=True)
                # print(f"Average number of words: {np.mean([len(x.split()) for x in data])}", flush=True)

                # define generative model which generates texts
                base_model, base_tokenizer = load_base_model_and_tokenizer(base_model_name, args.openai_model, args.human_dataset_name, args.cache_dir)

                text_generator = model_text_generator(dataset_name=args.human_dataset_name, base_model=base_model, base_tokenizer=base_tokenizer, cache_dir=data_path)
                text_generator.move_base_model_to_gpu()
                final_data = generate_model_written_dataset(data, text_generator)
                del text_generator
            else:
                # load a user pre-processed dataset
                final_data = load_preprocessed_dataset(args.human_dataset_name, synthetic_dataset_name, data_path, args.user_data_path)

            with open(os.path.join(data_path, 'experiment_data.json'), 'w') as f:
                print(f"Writing raw data to {os.path.join(data_path, 'experiment_data.json')}")
                json.dump(final_data, f)
        print('-----')
        sys.exit()

        if args.detect_method == 'detect_gpt':
            base_model, base_tokenizer = load_base_model_and_tokenizer(base_model_name, args.openai_model, args.human_dataset_name, args.cache_dir)

            if args.openai_model is not None:
                assert args.openai_key is not None, "Must provide OpenAI API key as --openai_key"
            else:
                base_model = base_model.to(device)

            if args.mask_filling_model_name:
                # mask filling t5 model
                print(f'Loading mask filling model {args.mask_filling_model_name}...', flush=True)
                mask_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(args.mask_filling_model_name, cache_dir=args.cache_dir)
                try:
                    n_positions = mask_model.config.n_positions
                except AttributeError:
                    n_positions = 512
            else:
                mask_model = None
                n_positions = 512

            mask_tokenizer = transformers.AutoTokenizer.from_pretrained(args.mask_filling_model_name, model_max_length=n_positions, cache_dir=args.cache_dir)

            cur_models = model_collector(base_model=base_model, base_tokenizer=base_tokenizer, mask_model=mask_model, mask_tokenizer=mask_tokenizer, device=device, openai_model=args.openai_model, openai_key=args.openai_key)
            cur_models.load_base_model()

            detect_method = DetectGPT(args.n_samples)
            detect_method.main_func(final_data, cur_models, results_path, args.mask_filling_model_name, args.batch_size)
        
        elif args.method == 'gpt2_detector':
            if args.pretrained_model == 'yes':
                pretrained_model = True

                import urllib.request

                assert args.pretrained_model_name in ['roberta-base', 'roberta-large'], "Must provide an effective pretrained model name, either 'roberta-base' or 'roberta-large'"

                if args.pretrained_model_name == 'roberta-base':
                    download_path = 'https://openaipublic.azureedge.net/gpt-2/detector-models/v1/detector-base.pt'
                    saved_path = args.cache_dir + '/detector-base.pt'
                else:
                    download_path = 'https://openaipublic.azureedge.net/gpt-2/detector-models/v1/detector-large.pt'
                    saved_path = args.cache_dir + '/detector-large.pt'
                if not os.path.exists(saved_path):
                    urllib.request.urlretrieve(download_path, saved_path)

                print(f'Loading checkpoint from {saved_path}', flush=True)
                para = torch.load(saved_path, map_location='cpu')

                classification_model = transformers.RobertaForSequenceClassification.from_pretrained(args.pretrained_model_name, cache_dir=args.cache_dir)
                classification_tokenizer = transformers.RobertaTokenizer.from_pretrained(args.pretrained_model_name, cache_dir=args.cache_dir)

                classification_model.load_state_dict(para['model_state_dict'], strict=False)
                classification_model.eval()
            else:
                pretrained_model = False

                classification_model = transformers.RobertaForSequenceClassification.from_pretrained(args.pretrained_model_name, cache_dir=args.cache_dir)
                classification_tokenizer = transformers.RobertaTokenizer.from_pretrained(args.pretrained_model_name, cache_dir=args.cache_dir)

            cur_models = model_collector(classification_model=classification_model, classification_tokenizer=classification_tokenizer, device=device, openai_model=args.openai_model, openai_key=args.openai_key)
            cur_models.load_classification_model()

            detect_method = GPT2_detector(pretrained_model, args.n_samples)
            if pretrained_model:
                detect_method.main_func(final_data, cur_models, results_path, args.batch_size)
            else:
                detect_method.train_func(cur_models, seed=args.seed, cache_dir=args.cache_dir, device='cuda', real_dataset='webtext', fake_dataset='xl-1542M-nucleus', data=final_data, results_path=results_path)
        
        elif args.method == 'gltr':
            if base_model is None:
                base_model, base_tokenizer = load_base_model_and_tokenizer(args.base_model_name, args.openai_model, args.dataset, args.cache_dir)

            if args.openai_model is not None:
                assert args.openai_key is not None, "Must provide OpenAI API key as --openai_key"
            else:
                base_model = base_model.to(device)

            cur_models = model_collector(base_model=base_model, base_tokenizer=base_tokenizer, device=device, openai_model=args.openai_model, openai_key=args.openai_key)
            cur_models.load_base_model()

            detect_method = GLTR(args.n_samples)
            detect_method.main_func(final_data, cur_models, results_path, args.batch_size)
        
        elif args.method == 'gpt_sentinel':
            if args.pretrained_model == 'yes':
                pretrained_model = True
                from collections import OrderedDict
                assert args.pretrained_model_name in ['roberta-sentinel', 't5-sentinel'], "Must provide an effective pretrained model name, either 'roberta-sentinel' or 't5-sentinel'"

                if args.pretrained_model_name == 'roberta-sentinel':
                    saved_path = args.cache_dir + '/roberta+mlp.base.0425.pt'
                else:
                    saved_path = args.cache_dir + '/t5.small.0422.pt'
                assert os.path.exists(saved_path), "Please download pretrained model from https://drive.google.com/drive/folders/17IPZUaJ3Dd2LzsS8ezkelCfs5dMDOluD into './cache_dir/' folder first"

                print(f'Loading checkpoint from {saved_path}', flush=True)
                para = torch.load(saved_path, map_location='cpu')

                if args.pretrained_model_name == 'roberta-sentinel':
                    classification_model = Roberta_Sentinel(args.cache_dir)
                    classification_tokenizer = transformers.RobertaTokenizer.from_pretrained('roberta-base', cache_dir=args.cache_dir)
                    classification_model.load_state_dict(para['model'], strict=False)
                else:
                    classification_model = transformers.T5ForConditionalGeneration.from_pretrained('t5-small', cache_dir=args.cache_dir)
                    classification_tokenizer = transformers.T5Tokenizer.from_pretrained('t5-small', cache_dir=args.cache_dir)

                    new_state_dict = OrderedDict()
                    for key, value in para['model'].items(): 
                        name = key.replace('t5_model.', '')
                        new_state_dict[name] = value 

                    classification_model.load_state_dict(new_state_dict, strict=False)
                classification_model.eval()
            else:
                pretrained_model = False

                if args.pretrained_model_name == 'roberta-sentinel':
                    classification_model = Roberta_Sentinel(args.cache_dir)
                    classification_tokenizer = transformers.RobertaTokenizer.from_pretrained('roberta-base', cache_dir=args.cache_dir)
                else:
                    classification_model = transformers.T5ForConditionalGeneration.from_pretrained('t5-small', cache_dir=args.cache_dir)
                    classification_tokenizer = transformers.T5Tokenizer.from_pretrained('t5-small', cache_dir=args.cache_dir)

            cur_models = model_collector(classification_model=classification_model, classification_tokenizer=classification_tokenizer, device=device, openai_model=args.openai_model, openai_key=args.openai_key)
            cur_models.load_classification_model()

            detect_method = GPT_Sentinel(pretrained_model, args.n_samples)
            if pretrained_model:
                detect_method.main_func(final_data, cur_models, results_path, args.batch_size, args.pretrained_model_name)
            else:
                detect_method.train_func(cur_models, cache_dir=args.cache_dir, device='cuda', data=final_data, results_path=results_path, pretrained_model_name=args.pretrained_model_name)

if __name__ == "__main__":
    main()  


