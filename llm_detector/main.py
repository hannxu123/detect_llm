import argparse

parser = argparse.ArgumentParser(description='Detect LLM generated texts')
parser.add_argument('--seed', type=int, default=0, help='seed')
parser.add_argument('--detect_method', default='roberta_base', type=str, help='name of detection method')
parser.add_argument('--pretrained_model_name', type=str, default='', help='name of a pretrained detector')

parser.add_argument('--full_dataset', default='hc3', type=str, help='name of dataset used for evaluation')
parser.add_argument('--human_dataset', type=str, default='', help='name of the human written dataset')
parser.add_argument('--synthetic_dataset', type=str, default='', help='name of the model generated dataset')
parser.add_argument('--generation_model_name', type=str, default='', help='name of the model used for generating text samples')
parser.add_argument('--generation_number', type=int, default=5000, help='number of text samples to generate')
parser.add_argument('--openai_model', type=str, default=None, help='name of OpenAI model')
parser.add_argument('--openai_key', type=str, default='', help='key for calling OpenAI API')
parser.add_argument('--cache_dir', type=str, default='./cache_dir', help='path of the cache folder')
parser.add_argument('--device', default='cuda', type=str, help='device used to execute')

parser.add_argument('--local', default='', type=str, help='the gpu number used on developing node.')

args = parser.parse_args()

import os
if args.local != '':
    os.environ['CUDA_VISIBLE_DEVICES'] = args.local

import numpy as np
import torch
import random
import transformers

import process_datasets
import evaluation
from methods.roberta_classifiers import *


def main():
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    print(f'Seed number is {args.seed}', flush=True)

    # create a cache folder to save models and datasets
    if not os.path.exists(args.cache_dir):
        os.makedirs(args.cache_dir)

    # load dataset
    print(f'Loading DATASET {args.full_dataset}...' , flush=True)
    # text data will be processed in a dictionary {original:[text1, text2, ...], sampled:[text1, text2, ...]}
    data = process_datasets.load(args.full_dataset, args.human_dataset, args.synthetic_dataset, args.cache_dir, args.generation_model_name, args.generation_number, args.openai_model)  

    # choose detection method
    if args.pretrained_model_name:
        method_name = args.pretrained_model_name
        classification_model, classification_tokenizer = load_availiable_detector(args.pretrained_model_name, args.cache_dir)
    elif args.detect_method == 'roberta_base' or args.detect_method == 'roberta_large' or args.detect_method == 'roberta_sentinel' or args.detect_method == 't5_sentinel':
        method_name = args.detect_method
        detect_process = roberta_classifiers(args.detect_method, args.full_dataset, args.cache_dir, args.device)
        classification_model, classification_tokenizer = detect_process.train_func(data)
    # elif args.detect_method == 'gpt_pat':
    #     detect_process = GPT_Pat(args.detect_method, args.full_dataset, args.cache_dir, args.device)
    #     classification_model, classification_tokenizer = detect_process.train_func(data)
    # elif args.detect_method == 'detect_gpt':
    #     DetectGPT(args)
    # elif args.detect_method == 'mpg':
    #     MPG(args)
    # elif args.detect_method == 'black_box_watermark':
    #     Black_box_watermark(args)
    cur_models = model_collector(classification_model=classification_model, classification_tokenizer=classification_tokenizer, device=args.device, openai_model=args.openai_model, openai_key=args.openai_key)
    #cur_models.load_classification_model()

    evaluation.detection_eval(data, cur_models, method_name)

if __name__ == "__main__":
    main()  


